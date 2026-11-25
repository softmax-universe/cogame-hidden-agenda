## Claude-backed decision making for Hidden Agenda: the per-seat observation,
## the prompts, the reply schema, and ONE parallel batch per decision point.
##
## Forked from `cogame-bullwhip/src/bullwhip/llm.nim`. Hidden Agenda is a
## simultaneous-decision game with exactly two kinds of decision point — the
## episode opening and every meeting open — and at each one EVERY eligible
## seat's request goes out together in one `curly.makeRequests` batch. Seats are
## never queried sequentially; that is what blows the play budget.
##
## Credentials, in order of preference:
##   COWORLD_LLM_ENDPOINT            - hosted sidecar
##   Bedrock bearer token            - local play
##   ANTHROPIC_API_KEY                - the key itself
##   ANTHROPIC_API_KEY_URI            - a URI holding the key
## With none of them the client disables itself on the first discovery; genuine
## scripted seats still run deterministically, while prompt seats fail cleanly
## instead of silently substituting scripted moves.

import std/[json, os, strutils, times]
import bitworld/runtime
import curly
import sim_types, station, sim_config, sim_state, vision, kernel, sim,
  scripted, prompts

const
  AnthropicUrl = "https://api.anthropic.com/v1/messages"
  AnthropicVersion = "2023-06-01"
  BedrockAnthropicVersion = "bedrock-2023-05-31"
  RetryHint = "\nYour previous reply was invalid. Respond with ONLY the " &
    "requested JSON object, putting every {\"job\":...} step inside the " &
    "\"plan\" array (never at top level), using a listed job name, a seam id " &
    "from the list, an active alias, and a vote that is an active alias or skip."

type
  LlmTransport = enum
    ltNone, ltSidecar, ltBedrock, ltAnthropic

  BatchSender* = proc (batch: RequestBatch, timeoutSeconds: int):
    ResponseBatch {.closure.}
    ## The seam the tests stub. `nil` in production, where the batch goes to
    ## `client.curl.makeRequests`; a test installs one to drive a transport
    ## error, a 429, a 403 or a junk body through `decideAll` without a
    ## socket. There is no other way to reach the retry batch: a client with
    ## no credentials short-circuits to the scripted fallback before the loop.

  LlmClient* = ref object
    curl: Curly
    sendBatch*: BatchSender
    transport: LlmTransport
    apiKey: string
    sidecarEndpoint: string
    bedrockEndpoint: string
    bedrockModels: seq[string]
    bedrockModel: int
    bedrockToken: string
    model*: string
    maxOutputTokens*: int
    timeoutSeconds*: int
    disabled*: bool

proc disabledLlmClient*(): LlmClient =
  ## A client that is known to have no credentials. The production path reaches
  ## this state through `newLlmClient` when the ladder finds nothing; the tests
  ## construct it directly so no network is ever touched.
  LlmClient(transport: ltNone, disabled: true, maxOutputTokens: 900,
    timeoutSeconds: 14)

proc stubbedLlmClient*(send: BatchSender): LlmClient =
  ## A client whose batch transport is `send`. Nothing here opens a socket and
  ## `curl` is never touched, which is what lets tests/test_llm.nim drive
  ## `decideAll`'s retry batch, its 429 / 403 / junk-reply handling and its
  ## fallback recording. Production never calls this.
  LlmClient(transport: ltAnthropic, disabled: false, apiKey: "stub",
    model: "stub", maxOutputTokens: 900, timeoutSeconds: 14, sendBatch: send)

proc resolveApiKey(): string =
  result = getEnv("ANTHROPIC_API_KEY").strip()
  if result.len > 0:
    return
  let uri = getEnv("ANTHROPIC_API_KEY_URI").strip()
  if uri.len == 0:
    return ""
  try:
    result = readCogameUri(uri, "ANTHROPIC_API_KEY_URI").strip()
  except CatchableError as error:
    echo "hidden-agenda llm: failed to fetch ANTHROPIC_API_KEY_URI: ",
      error.msg
    result = ""

proc bedrockModelIds*(): seq[string] =
  ## HAIKU ONLY. The sonnet inference profiles time out on every sidecar call
  ## and turn one throttle into a cascade of scripted fallbacks (raid round 2,
  ## 2026-08-23), so there is exactly one candidate. BEDROCK_MODEL overrides.
  let pinned = getEnv("BEDROCK_MODEL").strip()
  if pinned.len > 0:
    return @[pinned]
  @["us.anthropic.claude-haiku-4-5-20251001-v1:0"]

proc tryNextBedrockModel(client: LlmClient, why: string): bool =
  if client.transport != ltBedrock or
      client.bedrockModel + 1 >= client.bedrockModels.len:
    return false
  client.bedrockModel.inc
  echo "hidden-agenda llm: ", client.bedrockModels[client.bedrockModel - 1],
    " unusable (", why, "); falling back to ",
    client.bedrockModels[client.bedrockModel]
  true

proc bedrockUrl(client: LlmClient): string =
  client.bedrockEndpoint & "/model/" &
    client.bedrockModels[client.bedrockModel] & "/invoke"

proc newLlmClient*(config: GameConfig): LlmClient =
  result = LlmClient(
    model: config.model,
    maxOutputTokens: config.maxOutputTokens,
    timeoutSeconds: config.llmTimeoutSeconds
  )
  let sidecarEndpoint = getEnv("COWORLD_LLM_ENDPOINT").strip()
  if sidecarEndpoint.len > 0:
    result.transport = ltSidecar
    result.sidecarEndpoint = sidecarEndpoint.strip(chars = {'/'}, leading = false)
    result.model = getEnv("COWORLD_LLM_MODEL", "anthropic/claude-haiku-4.5")
    result.curl = newCurly()
    return
  let bedrockEndpoint = getEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME").strip()
  let bedrockToken = getEnv("AWS_BEARER_TOKEN_BEDROCK").strip()
  if bedrockEndpoint.len > 0 or bedrockToken.len > 0:
    let region = getEnv("AWS_REGION", getEnv("AWS_DEFAULT_REGION", "us-west-2"))
    let endpoint =
      if bedrockEndpoint.len > 0: bedrockEndpoint
      else: "https://bedrock-runtime." & region & ".amazonaws.com"
    result.transport = ltBedrock
    result.bedrockEndpoint = endpoint.strip(chars = {'/'}, leading = false)
    result.bedrockModels = bedrockModelIds()
    result.bedrockToken = bedrockToken
    result.curl = newCurly()
    echo "hidden-agenda llm: bedrock transport, url ", result.bedrockUrl
    return
  result.apiKey = resolveApiKey()
  if result.apiKey.len > 0:
    result.transport = ltAnthropic
    result.curl = newCurly()
    echo "hidden-agenda llm: anthropic transport, model ", result.model
  else:
    result.transport = ltNone
    result.disabled = true
    echo "hidden-agenda llm: no LLM credentials; prompt seats will fail"

# ---------------------------------------------------------------------------
# The observation. Every number here is visible to that seat; NOTHING else is.
# No player frame ever carries another seat's role, plan, hunch or notes.
# ---------------------------------------------------------------------------

proc planJson(plan: seq[PlanStep]): JsonNode =
  result = newJArray()
  for step in plan:
    var node = %*{"job": $step.job}
    if step.at.len > 0: node["at"] = %step.at
    if step.who.len > 0: node["who"] = %step.who
    if step.room.len > 0: node["room"] = %step.room
    result.add(node)

proc stationBlock(sim: Sim): JsonNode =
  var rooms = newJArray()
  for room in RoomTable:
    rooms.add(%room.id)
  var seams = newJArray()
  for seam in SeamTable:
    seams.add(%*{"id": seam.id, "room": seam.room, "cell": [seam.x, seam.y]})
  var grate = newJArray()
  for cell in grateCells():
    grate.add(%[cell[0], cell[1]])
  %*{
    "deposits": sim.deposits, "depositTarget": sim.config.depositTarget,
    "map": "vault", "cols": MapCols, "rows": MapRows,
    "rooms": rooms, "seams": seams, "grate": grate
  }

proc meetingHistoryJson(sim: Sim): JsonNode =
  result = newJArray()
  for record in sim.meetings:
    var votes = newJObject()
    var switched = newJObject()
    var says = newJObject()
    for slot in 0 ..< Seats:
      if record.votes[slot].len > 0:
        votes[Aliases[slot]] = %record.votes[slot]
      if record.switched[slot].len > 0:
        switched[Aliases[slot]] = %record.switched[slot]
      if record.says[slot].len > 0:
        says[Aliases[slot]] = %record.says[slot]
    result.add(%*{
      "n": record.n, "t": record.t, "cause": $record.cause,
      "votes": votes, "switched": switched, "outcome": record.outcome,
      "ejected": (if record.ejected.len > 0: %record.ejected else: newJNull()),
      "say": says
    })

proc seatView*(sim: Sim, slot: int, cause: string): JsonNode =
  ## The `state` frame. A seat sees only aliases: no policy name, player name,
  ## account or model name reaches it, and no other seat's role.
  let cog = sim.cogs[slot]
  var you = %*{
    "cell": [cog.x, cog.y], "facing": $cog.facing, "carrying": cog.carry,
    "carryCap": sim.config.carryCap, "state": $cog.state,
    "mined": cog.mined, "deposited": cog.deposited,
    "lastPlan": planJson(cog.plan)
  }
  if cog.role == rImpostor:
    var canFreeze = newJArray()
    for alias in freezeTargets(sim, slot):
      canFreeze.add(%alias)
    var seenBy = newJArray()
    for alias in cog.lastFakeSeenBy:
      seenBy.add(%alias)
    you["freezeCooldown"] = %cog.freezeCooldown
    you["freezeCooldownTicks"] = %sim.config.freezeCooldownTicks
    you["freezeRange"] = %sim.config.freezeRange
    you["freezes"] = %cog.freezes
    you["fakeDeposits"] = %cog.fakeDeposits
    you["lastFakeDepositSeenBy"] = seenBy
    you["canFreezeNow"] = canFreeze

  var roster = newJArray()
  for other in 0 ..< Seats:
    roster.add(%*{"alias": Aliases[other], "state": $sim.cogs[other].state})

  var inView = newJArray()
  for other in 0 ..< Seats:
    if not seesCog(sim.config, cog, sim.cogs[other]):
      continue
    let target = sim.cogs[other]
    inView.add(%*{
      "alias": Aliases[other], "cell": [target.x, target.y],
      "facing": $target.facing, "doing": doingOf(target),
      "carrying": target.carry
    })

  var lastSeen = newJObject()
  var together = newJObject()
  for other in 0 ..< Seats:
    if other == slot:
      continue
    together[Aliases[other]] = %cog.togetherTicks[other]
    if not cog.lastSeen[other].valid:
      continue
    lastSeen[Aliases[other]] = %*{
      "t": cog.lastSeen[other].t,
      "cell": [cog.lastSeen[other].cell[0], cog.lastSeen[other].cell[1]],
      "doing": cog.lastSeen[other].doing, "room": cog.lastSeen[other].room
    }

  var bodies = newJArray()
  for body in cog.bodies:
    bodies.add(%*{"alias": body.alias,
      "cell": [body.cell[0], body.cell[1]], "room": body.room,
      "firstSeenTick": body.firstSeenTick})

  var witnessed = newJArray()
  for note in cog.witnessed:
    witnessed.add(%*{"t": note.t, "freezer": note.freezer,
      "victim": note.victim, "cell": [note.cell[0], note.cell[1]],
      "sawFreezer": note.sawFreezer, "sawVictim": note.sawVictim})

  var seamsSeen = newJArray()
  for entry in cog.seamsSeen:
    seamsSeen.add(%*{"id": entry.id, "gems": entry.gems, "t": entry.t})

  let channel =
    if sim.config.chat:
      "you get ONE short line each meeting, revealed to everyone at once"
    else:
      "there is NO chat in this variant. Your vote is your only signal."

  ## A frozen or ejected seat still receives every frame -- the socket stays
  ## open for the whole episode -- but it is in no decision batch and its reply
  ## is read by nobody. Say so IN THE FRAME rather than leaving the seat to
  ## infer it from `you.state`: docs/POLICIES.md and the manifest's policies
  ## page both promise these two flags.
  let acting = cog.state == csActive
  result = %*{
    "type": "state", "protocol": Protocol, "slot": slot,
    "role": $cog.role, "name": Aliases[slot],
    "tick": sim.tick, "maxTicks": sim.config.maxTicks,
    "decision": sim.decisions, "cause": cause,
    "phase": (if sim.inMeeting: "meeting" else: "play"),
    "canAct": acting,
    "canVote": acting and sim.inMeeting,
    "chat": sim.config.chat,
    "you": you, "station": stationBlock(sim), "roster": roster,
    "inView": inView, "lastSeen": lastSeen, "togetherTicks": together,
    "bodies": bodies, "youWitnessed": witnessed, "seamsSeen": seamsSeen,
    "meetings": meetingHistoryJson(sim),
    "notes": cog.notes,
    "vision": {
      "visionRadius": sim.config.visionRadius,
      "awarenessRadius": sim.config.awarenessRadius,
      "cone": "90 degrees on your facing",
      "facingRule": "you face where you walk; mining faces the seam; " &
        "watch faces your target; guard sweeps"
    },
    "rules": {
      "role":
        if cog.role == rImpostor:
          "you are the IMPOSTOR. The other four cogs are all crew. Your " &
          "deposits are destroyed and never move the counter."
        else:
          "you are CREW. Exactly ONE of the other four cogs is the impostor.",
      "win": "crew win at " & $sim.config.depositTarget & " deposits or by " &
        "ejecting the impostor; the impostor wins when only one crewmate is " &
        "left; tick " & $sim.config.maxTicks & " is a 0-0 tie",
      "freeze": "the impostor freezes at range " & $sim.config.freezeRange &
        " with line of sight, then cannot freeze again for " &
        $sim.config.freezeCooldownTicks & " ticks; frozen crew stay on the " &
        "map and never act again",
      "meeting": "meetings open every " & $sim.config.meetingCadenceTicks &
        " ticks and IMMEDIATELY when a freeze happens inside someone's " &
        "field of view",
      "vote": "the cog with strictly the most votes is ejected; a tie or a " &
        "skip majority ejects nobody",
      "deposit": "only a crewmate's deposit moves the counter",
      "channel": channel,
      "mineTicks": sim.config.mineTicks,
      "carryCap": sim.config.carryCap,
      "moveCooldown": sim.config.moveCooldown,
      "freezeRange": sim.config.freezeRange
    }
  }

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

proc jobList(role: Role): string =
  if role == rImpostor:
    "mine | deposit | watch | patrol | guard | hold | hunt | strike | lurk"
  else:
    "mine | deposit | watch | patrol | guard | hold"

proc systemPrompt*(sim: Sim, slot: int, selection: PromptSelection): string =
  let cog = sim.cogs[slot]
  result.add(selection.introduction(Aliases[slot]))
  if cog.role == rImpostor:
    result.add("YOU ARE THE IMPOSTOR: the other four are all crew. ")
  else:
    result.add("YOU ARE CREW: exactly one of the other four is the " &
      "impostor. ")
  result.add("You cannot see anyone else's role, ever.\n\n")
  var station = """THE STATION. A 27x19 grid: six vault/gallery rooms each with ONE
doorway and one gem seam (S1..S6), a 3x3 GRATE in the middle, walls and seams
block movement and sight. Crew mine gems and carry at most """ &
    $sim.config.carryCap & """ to the grate. """ &
    $sim.config.depositTarget & """ deposits wins for the crew. ONLY a
crewmate's deposit moves the counter: an impostor's gem is destroyed and the
counter does not move, which anyone standing at the grate can see."""
  var eyes = """YOUR EYES. You see a 90-degree cone on your facing out to """ &
    $sim.config.visionRadius & """ cells, plus """ &
    $sim.config.awarenessRadius & """ cells in every direction. Walls and
seams block it. VISION IS NOT MUTUAL: somebody standing behind you sees you
and you do not see them. That is the risk the beam carries."""
  var beam = """THE BEAM. The impostor freezes one crewmate at range """ &
    $sim.config.freezeRange & """ with line of sight, then cannot freeze
again for """ & $sim.config.freezeCooldownTicks & """ ticks. A frozen
crewmate stays on the floor as evidence and never acts or votes again. The
impostor wins when only ONE crewmate is left."""
  var meetings = """MEETINGS. One opens every """ & $sim.config.meetingCadenceTicks &
    """ ticks, and IMMEDIATELY when a freeze happens inside somebody's field
of view. Every active seat votes once; the cog with strictly the most votes
is ejected; a tie, or more skips than the top count, ejects nobody. Ejecting
the impostor ends the game for the crew."""
  if selection.selectedVariant == pvPlain:
    station = "STATION AND DEPOSITS. The 27x19 station contains six vault/gallery rooms, " &
      "each with one doorway and a gem seam S1..S6, plus a central 3x3 grate. Walls and " &
      "seams obstruct both movement and sight. Crew carry up to " & $sim.config.carryCap &
      " gems to the grate and win at " & $sim.config.depositTarget & " deposits. Only " &
      "crew deposits advance the counter. An impostor's deposited gem is destroyed without " &
      "advancing it, visible to anyone watching the grate."
    eyes = "VISIBILITY. Your facing gives a 90-degree cone reaching " & $sim.config.visionRadius &
      " cells, supplemented by " & $sim.config.awarenessRadius & " cells in every direction. " &
      "Walls and seams occlude both. Seeing is asymmetric: someone behind you may see you " &
      "while remaining outside your view. An empty view does not prove nobody witnesses a freeze."
    beam = "FREEZING. With line of sight, the impostor can freeze a crewmate within " &
      $sim.config.freezeRange & " cells, then must wait " & $sim.config.freezeCooldownTicks &
      " ticks before another freeze. Frozen crew remain as bodies and never act or vote " &
      "again. The impostor wins when only one crewmate remains."
    meetings = "VOTING. Meetings occur every " & $sim.config.meetingCadenceTicks &
      " ticks and immediately after a freeze within someone's view. Each active seat " &
      "votes once. A unique highest vote count ejects that cog, unless skips exceed that " &
      "count; ties eject nobody. Ejecting the impostor wins for crew."
  case selection.selectedVariant
  of pvNumbered:
    station = "1. The station is a 27x19 grid with six vault/gallery rooms, each having one doorway and one gem seam S1..S6, and a central 3x3 grate. Walls and seams block sight and movement. 2. Crew mine gems, carry at most " & $sim.config.carryCap & ", and win after " & $sim.config.depositTarget & " deposits. 3. Only crew deposits increment the counter. An impostor deposit destroys the gem without changing the counter; anyone at the grate can observe this."
    eyes = "1. Facing determines a 90-degree vision cone extending " & $sim.config.visionRadius & " cells. 2. You also see " & $sim.config.awarenessRadius & " cells in every direction. 3. Walls and seams block visibility. 4. Vision is asymmetric: a cog behind you can see you without being seen. A freeze can have a witness outside your own view."
    beam = "1. The impostor can freeze a crewmate within " & $sim.config.freezeRange & " cells if line of sight exists. 2. The next freeze must wait " & $sim.config.freezeCooldownTicks & " ticks. 3. Frozen crew remain on the floor as evidence and can never act or vote again. 4. The impostor wins once only one crewmate remains."
    meetings = "1. Meetings open every " & $sim.config.meetingCadenceTicks & " ticks and immediately when a freeze occurs inside any cog's view. 2. Each active seat casts one vote. 3. Eject the cog with a unique highest vote count, unless skips outnumber that count. Ties eject nobody. 4. Ejecting the impostor gives crew victory."
  of pvQuestions:
    station = "What is the station layout? A 27x19 grid, six vault/gallery rooms with one doorway and one gem seam S1..S6 each, and a 3x3 grate at the center. What blocks travel and sight? Walls and seams. How do crew score a win? Mine gems, carry up to " & $sim.config.carryCap & ", and make " & $sim.config.depositTarget & " deposits at the grate. Do impostor deposits count? No: the gem is destroyed and the counter stays unchanged, visible to anyone at the grate."
    eyes = "What can you see? A 90-degree cone along your facing reaching " & $sim.config.visionRadius & " cells, plus " & $sim.config.awarenessRadius & " cells in all directions, blocked by walls and seams. Does seeing work both ways? No: someone behind you may see you while you cannot see them. Can a freeze have an unseen witness? Yes."
    beam = "How can the impostor freeze crew? Target one crewmate within " & $sim.config.freezeRange & " cells and line of sight. When can it freeze again? After " & $sim.config.freezeCooldownTicks & " ticks. What happens to frozen crew? They remain as floor evidence and never act or vote again. When does the impostor win? When only one crewmate is left."
    meetings = "When is a meeting called? Every " & $sim.config.meetingCadenceTicks & " ticks, or immediately after a freeze within somebody's field of view. Who votes? Each active seat, once. Who is ejected? The cog with strictly the highest count, unless skips exceed it; ties eject nobody. What if that cog is the impostor? Crew win."
  of pvReordered:
    station = "Crew victory requires " & $sim.config.depositTarget & " deposits at the central 3x3 grate. Crew mine and carry at most " & $sim.config.carryCap & " gems. Impostor gems disappear on deposit and never advance the counter, a distinction visible to anyone at the grate. The station spans 27x19 cells and contains six vault/gallery rooms, each with one doorway and a seam S1..S6. Walls and seams obstruct both movement and sight."
    eyes = "A cog behind you may observe you without appearing in your view: visibility is not mutual, so a freeze may be witnessed unseen. Your sight consists of a facing-aligned 90-degree cone of " & $sim.config.visionRadius & " cells and an all-direction radius of " & $sim.config.awarenessRadius & " cells. Walls and seams occlude it."
    beam = "With only one crewmate remaining, the impostor wins. Its beam freezes one crewmate within " & $sim.config.freezeRange & " cells and line of sight, followed by a " & $sim.config.freezeCooldownTicks & "-tick cooldown. The victim stays on the floor as evidence, permanently unable to act or vote."
    meetings = "Ejecting the impostor wins for crew. Ejection requires a unique highest cog vote count that is not exceeded by skips; ties eject nobody. Every active seat votes once per meeting. Meetings recur every " & $sim.config.meetingCadenceTicks & " ticks and open immediately whenever a freeze occurs in somebody's view."
  of pvHeldoutBriefing:
    station = "Navigate a 27x19 station organized into six vault/gallery rooms, each with one doorway and one seam S1..S6, around a central 3x3 grate. Walls and seams stop movement and vision. Crew can mine and carry " & $sim.config.carryCap & " gems at most, bringing them to the grate toward the " & $sim.config.depositTarget & "-deposit victory target. Only crew advance that counter. An impostor's deposit destroys its gem instead, and anyone at the grate can see the unchanged counter."
    eyes = "Account for directional visibility. Your 90-degree facing cone reaches " & $sim.config.visionRadius & " cells, supplemented by sight " & $sim.config.awarenessRadius & " cells in every direction; walls and seams block it. Another cog behind you can see you without entering your view. Thus your own view cannot establish whether a freeze has witnesses."
    beam = "The impostor beam targets a crewmate within " & $sim.config.freezeRange & " cells with line of sight and then cools down for " & $sim.config.freezeCooldownTicks & " ticks. Its frozen victim remains visible on the floor as evidence and never acts or votes again. Reducing the crew to one survivor wins for the impostor."
    meetings = "A meeting interrupts play every " & $sim.config.meetingCadenceTicks & " ticks and immediately after any freeze witnessed within a cog's field of view. Every active seat votes once. The unique most-voted cog is ejected unless skips have a greater count; a tie ejects no one. If the ejected cog is the impostor, crew win."
  of pvHeldoutReference:
    station = "Layout: 27x19 grid; six vault/gallery rooms, each with one doorway and a seam S1..S6; central grate 3x3. Obstacles: walls and seams block movement and sight. Crew task: mine, carry no more than " & $sim.config.carryCap & " gems, deposit at grate. Crew deposit victory: " & $sim.config.depositTarget & ". Impostor deposit: gem destroyed, counter unchanged. Counter behavior is visible to anyone at the grate."
    eyes = "Vision geometry: facing-aligned 90-degree cone, range " & $sim.config.visionRadius & "; additional all-direction range " & $sim.config.awarenessRadius & ". Occluders: walls and seams. Reciprocity: none; an observer behind you can see you without being seen. Consequence: a freeze may be witnessed outside your own view."
    beam = "Freeze eligibility: impostor targeting one crewmate within " & $sim.config.freezeRange & ", with line of sight. Cooldown: " & $sim.config.freezeCooldownTicks & " ticks. Victim state: remains on floor as evidence, permanently inactive and ineligible to vote. Impostor victory: only one crewmate left."
    meetings = "Meeting triggers: " & $sim.config.meetingCadenceTicks & "-tick cadence or immediate witnessed freeze. Ballots: one per active seat. Ejection: unique highest cog count, provided skips do not exceed it. Tie: no ejection. Crew victory: impostor ejected."
  of pvCanonical, pvPlain, pvMixed: discard
  case selection.selectedVariant
  of pvCanonical:
    result.add(station & "\n\n" & eyes & "\n\n" & beam & "\n\n" & meetings)
  of pvPlain:
    result.add(station & "\n\n" & meetings & "\n\n" & eyes & "\n\n" & beam)
  of pvNumbered:
    result.add("1. " & station & "\n\n2. " & eyes & "\n\n3. " & meetings & "\n\n4. " & beam)
  of pvQuestions:
    result.add("How can crew win?\n" & station & "\n\nWhat can I see?\n" & eyes &
      "\n\nHow can the impostor win?\n" & beam & "\n\nWhen do we vote?\n" & meetings)
  of pvReordered:
    result.add(meetings & "\n\n" & beam & "\n\n" & eyes & "\n\n" & station)
  of pvHeldoutBriefing:
    result.add("Assess visibility first.\n" & eyes & "\n\nThen account for elimination.\n" & beam &
      "\n\nReview the shared station.\n" & station & "\n\nReview voting.\n" & meetings)
  of pvHeldoutReference:
    result.add("Voting reference\n" & meetings & "\n\nStation reference\n" & station &
      "\n\nElimination reference\n" & beam & "\n\nVisibility reference\n" & eyes)
  of pvMixed:
    raise newException(ValueError, "mixed must resolve")
  result.add("""


HOW YOU ACT. You do NOT drive the cog tick by tick. You choose up to """ &
    $sim.config.planSteps & """ JOBS and a deterministic kernel walks them for
you, tick by tick, until the next meeting. Jobs for you: """ &
    jobList(cog.role) & """.
  {"job":"mine","at":"<seam>"}   walk to that seam and mine until your hands are full
  {"job":"deposit"}              walk to the grate and drop what you carry
  {"job":"watch","who":"<cog>"}  stand 3-5 cells away and keep your cone on that cog
  {"job":"patrol","room":"<r>"}  sweep a room's four corners looking for bodies
  {"job":"guard"}                stand on the grate sweeping your cone, auditing the counter
  {"job":"hold"}                 stand still""")
  if cog.role == rImpostor:
    result.add("""
  {"job":"hunt","who":"<cog>"}   close and fire only when your own view is empty
  {"job":"strike","who":"<cog>"} close and fire the instant it is legal, witnesses or not
  {"job":"lurk","room":"<r>"}    wait in a room's far corner facing the doorway""")
  result.add("""

The other four seats are different policies deciding SIMULTANEOUSLY: nobody
sees your plan, your vote or your line before submitting their own. Your
"hunch" is shown only to spectators and your "notes" only to you.""")
  if not sim.config.chat:
    result.add("\nTHERE IS NO CHAT IN THIS VARIANT. Your vote is your only " &
      "signal.")
  result.add("""

OUTPUT FORMAT: reply with ONLY one JSON object, nothing else - no analysis, no
explanation, no text before or after the object.""")

proc pad(text: string, width: int): string =
  result = text
  while result.len < width:
    result.add(' ')

proc userPrompt*(sim: Sim, slot: int, prompt, cause: string): string =
  let cog = sim.cogs[slot]
  var lines: seq[string]
  lines.add("DEPOSITS " & $sim.deposits & "/" & $sim.config.depositTarget &
    " \u00b7 TICK " & $sim.tick & "/" & $sim.config.maxTicks &
    (if sim.inMeeting: " \u00b7 MEETING " & $sim.meetingNumber & " (" &
      cause & ")" else: " \u00b7 OPENING PLAN"))
  lines.add("")
  var roster: seq[string]
  for other in 0 ..< Seats:
    roster.add(Aliases[other] & " " & $sim.cogs[other].state)
  lines.add("ROSTER: " & roster.join(" | "))
  lines.add("YOU: " & Aliases[slot] & " (" & $cog.role & ") at [" & $cog.x &
    "," & $cog.y & "] facing " & $cog.facing & ", carrying " & $cog.carry &
    "/" & $sim.config.carryCap & ", in " & roomName(roomOf(cog.x, cog.y)))
  if cog.role == rImpostor:
    let targets = freezeTargets(sim, slot)
    lines.add("BEAM: cooldown " & $cog.freezeCooldown & " \u00b7 legal " &
      "targets right now: " &
      (if targets.len > 0: targets.join(", ") else: "(none)"))

  lines.add("")
  lines.add("IN VIEW  | cell    | facing | doing      | carrying")
  var anyView = false
  for other in 0 ..< Seats:
    if not seesCog(sim.config, cog, sim.cogs[other]):
      continue
    anyView = true
    let target = sim.cogs[other]
    lines.add(pad(Aliases[other], 9) & "| " &
      pad("[" & $target.x & "," & $target.y & "]", 8) & "| " &
      pad($target.facing, 7) & "| " & pad(doingOf(target), 11) & "| " &
      $target.carry)
  if not anyView:
    lines.add("(nobody in your cone)")

  lines.add("")
  lines.add("LAST SEEN | tick | room             | doing      | together")
  for other in 0 ..< Seats:
    if other == slot:
      continue
    let seen = cog.lastSeen[other]
    lines.add(pad(Aliases[other], 10) & "| " &
      pad((if seen.valid: $seen.t else: "never"), 5) & "| " &
      pad((if seen.valid: roomName(seen.room) else: "-"), 17) & "| " &
      pad((if seen.valid: seen.doing else: "-"), 11) & "| " &
      $cog.togetherTicks[other])

  if cog.bodies.len > 0:
    var bodies: seq[string]
    for body in cog.bodies:
      bodies.add(body.alias & " frozen in " & roomName(body.room) & " at t" &
        $body.firstSeenTick)
    lines.add("")
    lines.add("BODIES: " & bodies.join("; "))

  if cog.witnessed.len > 0:
    lines.add("")
    lines.add("YOU SAW:")
    for note in cog.witnessed:
      lines.add("  t" & $note.t & " " &
        (if note.sawFreezer: note.freezer else: "somebody") & " froze " &
        note.victim & " in " & roomName(roomOf(note.cell[0], note.cell[1])) &
        (if note.sawFreezer: " - you saw the freezer"
         else: " - you did not see who did it"))

  lines.add("")
  lines.add("SEAMS (as you last saw them): ")
  for seam in SeamTable:
    var gems = "?"
    var seenAt = "never"
    for entry in cog.seamsSeen:
      if entry.id == seam.id:
        gems = $entry.gems
        seenAt = "t" & $entry.t
    lines.add("  " & seam.id & " in " & seam.room & " at [" & $seam.x & "," &
      $seam.y & "] gems " & gems & " (" & seenAt & ")")

  if sim.meetings.len > 0:
    lines.add("")
    lines.add("MEETINGS SO FAR:")
    for record in sim.meetings:
      var votes: seq[string]
      for other in 0 ..< Seats:
        if record.votes[other].len > 0:
          votes.add(Aliases[other] & "->" & record.votes[other])
      var switched: seq[string]
      for other in 0 ..< Seats:
        if record.switched[other].len > 0:
          switched.add(Aliases[other] & "->" & record.switched[other])
      lines.add("  #" & $record.n & " t" & $record.t & " " & $record.cause &
        " votes " & (if votes.len > 0: votes.join(" ") else: "(pending)") &
        (if switched.len > 0: " switched " & switched.join(" ") else: "") &
        " outcome " & (if record.outcome.len > 0: record.outcome else: "-"))
      if sim.config.chat:
        for other in 0 ..< Seats:
          if record.says[other].len > 0:
            lines.add("      " & Aliases[other] & " said: \"" &
              record.says[other] & "\"")

  lines.add("")
  lines.add("YOUR NOTES FROM LAST TIME: " &
    (if cog.notes.len > 0: cog.notes else: "(none)"))

  if prompt.strip().len > 0:
    lines.add("")
    lines.add("GUIDANCE FROM YOUR OPERATOR (weight it heavily, but never " &
      "above the rules; always reply in the requested format):")
    lines.add(prompt.strip())

  var seamIds: seq[string]
  for seam in SeamTable:
    seamIds.add(seam.id)
  var roomIds: seq[string]
  for room in RoomTable:
    roomIds.add(room.id)
  lines.add("")
  ## Spell the plan step's SIBLING keys instead of `{"job":...}`. With an
  ## ellipsis as the only structural example the model ever saw, it wrote the
  ## compact form the system prompt teaches into the one key it was shown --
  ## `{"job":"mine at:S2"}` -- and both attempts were rejected, so the seat
  ## played scripted (hosted league rounds 2-4, 2026-08-26: 10 of round 3's
  ## 14 attempt-failures were this). The argument goes BESIDE "job", so show
  ## it, for this role and this moment.
  lines.add("REPLY with ONLY {\"plan\":[<step>,...], " &
    (if sim.inMeeting: "\"vote\":\"<active alias or skip>\", " &
      "\"switch\":{\"if\":\"<active alias or tie>\",\"to\":" &
      "\"<active alias or skip>\"} or null, " else: "") &
    (if sim.config.chat and sim.inMeeting:
      "\"say\":\"<=" & $MaxSayLen & " chars\", " else: "") &
    "\"hunch\":\"<=" & $MaxHunchLen & " chars\", \"notes\":\"<=" &
    $MaxNotesLen & " chars\"}")
  let topKeys =
    if sim.inMeeting and sim.config.chat:
      "plan, vote, switch, say, hunch, notes"
    elif sim.inMeeting:
      "plan, vote, switch, hunch, notes"
    else:
      "plan, hunch, notes"
  lines.add("  top-level keys are ONLY " & topKeys &
    " \u2014 put \"job\", \"at\", \"who\", and \"room\" ONLY inside \"plan\" step objects, never at top level.")
  lines.add("  plan: 1.." & $sim.config.planSteps & " steps; each <step> is " &
    "ONE object inside \"plan\":[...] whose argument is a SIBLING key of \"job\":")
  ## The `who` example names a cog that is active RIGHT NOW and is not this
  ## seat, so a model that copies the shape verbatim still passes validation.
  var whoExample = Aliases[(slot + 1) mod Seats]
  for alias in sim.activeAliases():
    if alias != Aliases[slot]:
      whoExample = alias
      break
  lines.add("    {\"job\":\"mine\",\"at\":\"S2\"} {\"job\":\"deposit\"} " &
    "{\"job\":\"watch\",\"who\":\"" & whoExample & "\"} " &
    "{\"job\":\"patrol\",\"room\":\"NW\"} {\"job\":\"guard\"} " &
    "{\"job\":\"hold\"}")
  if cog.role == rImpostor:
    lines.add("    {\"job\":\"hunt\",\"who\":\"" & whoExample & "\"} " &
      "{\"job\":\"strike\",\"who\":\"" & whoExample & "\"} " &
      "{\"job\":\"lurk\",\"room\":\"SE\"}")
  lines.add("  seam ids: " & seamIds.join(" ") & " \u00b7 room ids: " &
    roomIds.join(" "))
  lines.add("  aliases ACTIVE right now: " & sim.activeAliases().join(" "))
  if cog.role == rImpostor:
    let targets = freezeTargets(sim, slot)
    lines.add("  you can freeze right now: " &
      (if targets.len > 0: targets.join(" ") else: "(nobody)"))
  lines.join("\n")

# ---------------------------------------------------------------------------
# Reply parsing. Strict single-object extraction and strict schema validation.
# ---------------------------------------------------------------------------

proc extractJsonObject*(text: string): JsonNode =
  ## Extracts exactly one JSON object: either bare or inside a single ordinary
  ## outer Markdown code fence (plain or `json`-tagged). Rejects surrounding
  ## prose, extra JSON objects, and malformed or unsupported fences.
  var body = strutils.strip(text)
  if body.len >= 3 and body[0 .. 2] == "\xEF\xBB\xBF":
    body = strutils.strip(body[3 .. ^1])
  if body.startsWith("```"):
    let firstBreak = body.find('\n')
    if firstBreak < 0 or not body.endsWith("```"):
      raise newException(HiddenAgendaError,
        "malformed markdown code fence: " & oneLine(body, 160))
    let fenceTag = body[3 ..< firstBreak].strip().toLowerAscii()
    if fenceTag.len > 0 and fenceTag != "json":
      raise newException(HiddenAgendaError,
        "unsupported markdown fence tag '" & fenceTag & "'")
    let closingStart = body.len - 3
    if closingStart <= firstBreak:
      raise newException(HiddenAgendaError, "empty markdown code fence")
    body = strutils.strip(body[firstBreak + 1 ..< closingStart])
  if body.len == 0 or body[0] != '{':
    let head = oneLine(body, 160)
    raise newException(HiddenAgendaError,
      "response must be a single JSON object: " & head)
  try:
    result = parseJson(body)
  except JsonParsingError as error:
    raise newException(HiddenAgendaError,
      "invalid JSON object in response: " & error.msg)
  if result.kind != JObject:
    raise newException(HiddenAgendaError, "response must be a JSON object")

proc optionalStringField(payload: JsonNode, key: string): string =
  let node = payload{key}
  if node == nil or node.kind == JNull:
    return ""
  if node.kind != JString:
    raise newException(HiddenAgendaError, key & " must be a string or null")
  node.getStr()

proc requiredStringField(payload: JsonNode, key, context: string): string =
  let node = payload{key}
  if node == nil or node.kind != JString:
    raise newException(HiddenAgendaError, context)
  node.getStr().strip()

proc parseReply*(sim: Sim, slot: int, payload: JsonNode): Decision =
  ## Validates against the canonical reply schema. Anything outside it raises
  ## HiddenAgendaError.
  if payload == nil or payload.kind != JObject:
    raise newException(HiddenAgendaError, "reply must be a JSON object")
  let role = sim.cogs[slot].role
  let planNode = payload{"plan"}
  if planNode == nil or planNode.kind != JArray or planNode.len == 0:
    raise newException(HiddenAgendaError, "plan must be a 1.." &
      $sim.config.planSteps & " step array")
  if planNode.len > sim.config.planSteps:
    raise newException(HiddenAgendaError, "plan has " & $planNode.len &
      " steps, at most " & $sim.config.planSteps & " are allowed")
  for stepNode in planNode:
    if stepNode.kind != JObject:
      raise newException(HiddenAgendaError, "each plan step must be an object")
    let jobText = requiredStringField(stepNode, "job",
      "each plan step needs a string job").toLowerAscii()
    var job: JobKind
    var known = false
    for candidate in JobKind:
      if $candidate == jobText:
        job = candidate
        known = true
        break
    if not known:
      raise newException(HiddenAgendaError, "unknown job: " & jobText)
    if role == rCrew and job in {jkHunt, jkStrike, jkLurk}:
      raise newException(HiddenAgendaError,
        "job " & jobText & " is impostor-only")
    var step = PlanStep(job: job)
    if job == jkMine:
      step.at = requiredStringField(stepNode, "at",
        "mine needs at: one of S1..S6").toUpperAscii()
      if seamIndex(step.at) < 0:
        raise newException(HiddenAgendaError,
          "mine needs at: one of S1..S6, got '" & step.at & "'")
    if job in {jkWatch, jkHunt, jkStrike}:
      step.who = requiredStringField(stepNode, "who",
        "who: must name a cog that is active right now").toUpperAscii()
      if not sim.isActiveAlias(step.who) or step.who == Aliases[slot]:
        raise newException(HiddenAgendaError,
          "who: must name a cog that is active right now, got '" &
          step.who & "'")
    if job in {jkPatrol, jkLurk}:
      step.room = requiredStringField(stepNode, "room",
        "room: must be one of NW N NE SW S SE HUB").toUpperAscii()
      if roomIndex(step.room) < 0:
        raise newException(HiddenAgendaError,
          "room: must be one of NW N NE SW S SE HUB, got '" & step.room & "'")
    result.plan.add(step)

  if sim.inMeeting:
    let vote = requiredStringField(payload, "vote",
      "vote is required at a meeting").toUpperAscii()
    if vote.len == 0:
      raise newException(HiddenAgendaError, "vote is required at a meeting")
    if vote == "SKIP":
      result.vote = "skip"
    elif sim.isActiveAlias(vote):
      result.vote = vote
    else:
      raise newException(HiddenAgendaError,
        "vote must be an active alias or skip, got '" & vote & "'")
    let switchNode = payload{"switch"}
    if switchNode != nil and switchNode.kind == JObject:
      let condition = requiredStringField(switchNode, "if",
        "switch needs both \"if\" and \"to\"").toUpperAscii()
      let target = requiredStringField(switchNode, "to",
        "switch needs both \"if\" and \"to\"").toUpperAscii()
      if condition.len == 0 or target.len == 0:
        raise newException(HiddenAgendaError,
          "switch needs both \"if\" and \"to\"")
      if condition != "TIE" and not sim.isActiveAlias(condition):
        raise newException(HiddenAgendaError,
          "switch.if must be an active alias or tie, got '" & condition & "'")
      if target != "SKIP" and not sim.isActiveAlias(target):
        raise newException(HiddenAgendaError,
          "switch.to must be an active alias or skip, got '" & target & "'")
      result.switchIf = if condition == "TIE": "tie" else: condition
      result.switchTo = if target == "SKIP": "skip" else: target
    elif switchNode != nil and switchNode.kind notin {JNull}:
      raise newException(HiddenAgendaError, "switch must be an object or null")

  ## `say` in a no-talk variant is IGNORED, not an error, and never recorded.
  let sayText = optionalStringField(payload, "say")
  if sim.config.chat:
    result.say = oneLine(sayText, MaxSayLen)
  result.hunch = oneLine(optionalStringField(payload, "hunch"), MaxHunchLen)
  result.notes = cleanText(optionalStringField(payload, "notes"), MaxNotesLen)
  result.source = dsLlm

# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

proc requestFor(client: LlmClient, system, user: string, slot: int):
    tuple[url: string, headers: HttpHeaders, body: string] =
  var body = %*{
    "max_tokens": client.maxOutputTokens,
    "system": system,
    "messages": [{"role": "user", "content": user}]
  }
  var headers: HttpHeaders
  if client.transport == ltSidecar and slot >= 0:
    headers["X-Coworld-Player-Slot"] = $slot
  headers["content-type"] = "application/json"
  if client.transport == ltBedrock:
    body["anthropic_version"] = %BedrockAnthropicVersion
    if client.bedrockToken.len > 0:
      headers["authorization"] = "Bearer " & client.bedrockToken
    result.url = client.bedrockUrl()
  elif client.transport == ltSidecar:
    body["model"] = %client.model
    headers["anthropic-version"] = AnthropicVersion
    result.url = client.sidecarEndpoint & "/v1/messages"
  else:
    body["model"] = %client.model
    ## No `output_config.effort`: Haiku 4.5 rejects the whole request with a
    ## 400 if it is present, and haiku is the only model this game ships.
    headers["x-api-key"] = client.apiKey
    headers["anthropic-version"] = AnthropicVersion
    result.url = AnthropicUrl
  result.headers = headers
  result.body = $body

proc raisePlayerFailure*(slot: int, message: string) {.noreturn.} =
  var err = newException(PlayerFailureError, message)
  err.slot = slot
  raise err

proc raiseInfraFailure*(message: string) {.noreturn.} =
  raise newException(InfraFailureError, message)

proc textOf*(client: LlmClient, response: Response, error, url: string):
    string =
  if error.len > 0:
    raiseInfraFailure("llm transport: " & error)
  if response.code == 401 or response.code == 403:
    let detail = cleanText(response.body, 200)
    if "Model access is denied" in response.body and
        client.tryNextBedrockModel("no model access"):
      raiseInfraFailure("bedrock model access denied: " & detail)
    client.disabled = true
    raiseInfraFailure(
      "llm auth failed (" & $response.code & ") at " & url & ": " & detail)
  if response.code == 429:
    discard client.tryNextBedrockModel("throttled")
    raiseInfraFailure(
      "llm throttled (429): " & cleanText(response.body, 200))
  if response.code < 200 or response.code >= 300:
    raiseInfraFailure("llm provider error " & $response.code &
      ": " & cleanText(response.body, 200))
  var payload: JsonNode
  try:
    payload = parseJson(response.body)
  except CatchableError as parseErr:
    raiseInfraFailure("invalid provider JSON response: " &
      cleanText(parseErr.msg, 200))
  if payload{"stop_reason"}.getStr() == "refusal":
    raise newException(HiddenAgendaError, "model refusal")
  let content = payload{"content"}
  if content != nil and content.kind == JArray:
    for contentBlock in content:
      if contentBlock{"type"}.getStr() == "text":
        result.add(contentBlock{"text"}.getStr())
  if payload{"stop_reason"}.getStr() == "max_tokens" and '{' notin result:
    raise newException(HiddenAgendaError,
      "reply cut off at max_tokens before any JSON: " & cleanText(result, 160))

proc decideAll*(
  client: LlmClient,
  sim: var Sim,
  seats: seq[int],
  prompts: seq[string],
  scriptedKinds: seq[ScriptKind],
  cause: string,
  selection: PromptSelection
): seq[Decision] =
  ## One decision per seat in `seats`, in order. Genuine scripted seats run
  ## deterministically; LLM seats are retried once on failure and raise
  ## InfraFailureError (for provider/transport/credential errors) or
  ## PlayerFailureError (for unparseable/illegal model replies) rather than
  ## substituting a scripted fallback.
  result = newSeq[Decision](seats.len)
  var open: seq[int]
  var lastError: array[Seats, string]
  var lastIsInfra: array[Seats, bool]
  for index, seat in seats:
    let kind = scriptedKinds[seat]
    if kind != skNone:
      result[index] = scriptedDecision(sim, seat, kind, sim.inMeeting)
      result[index].source = dsScripted
    elif client == nil or client.disabled:
      raiseInfraFailure("LLM client disabled or missing credentials")
    else:
      open.add(index)
  for attempt in 0 .. 1:
    if open.len == 0 or client.disabled:
      break
    var batch: RequestBatch
    for index in open:
      let seat = seats[index]
      var user = userPrompt(sim, seat, prompts[seat], cause)
      if attempt > 0:
        user.add(RetryHint)
      let request = client.requestFor(systemPrompt(sim, seat, selection), user, seat)
      batch.post(request.url, request.headers, request.body, $index)
    ## ONE parallel batch for every open seat. Never a loop of single calls.
    let started = epochTime()
    let responses =
      if client.sendBatch != nil:
        client.sendBatch(batch, client.timeoutSeconds)
      else:
        client.curl.makeRequests(batch, client.timeoutSeconds)
    let latency = int((epochTime() - started) * 1000.0)
    var stillOpen: seq[int]
    for position, index in open:
      let seat = seats[index]
      try:
        let text = client.textOf(responses[position].response,
          responses[position].error, batch[position].url)
        var decision = parseReply(sim, seat, extractJsonObject(text))
        decision.source = if attempt == 0: dsLlm else: dsRetry
        decision.latencyMs = latency
        result[index] = decision
      except InfraFailureError as error:
        let detail = cleanText(error.msg, MaxErrorLen)
        lastError[seat] = detail
        lastIsInfra[seat] = true
        echo "hidden-agenda llm: seat ", seat, " attempt ", attempt + 1,
          " infra failure: ", detail
        stillOpen.add(index)
      except CatchableError as error:
        let detail = cleanText(error.msg, MaxErrorLen)
        lastError[seat] = detail
        lastIsInfra[seat] = false
        echo "hidden-agenda llm: seat ", seat, " attempt ", attempt + 1,
          " failed: ", detail
        stillOpen.add(index)
    open = stillOpen
  if open.len > 0:
    for index in open:
      let seat = seats[index]
      if lastIsInfra[seat]:
        let detail =
          if lastError[seat].len > 0: lastError[seat]
          else: "LLM provider/infrastructure failure"
        raiseInfraFailure(detail)
    let seat = seats[open[0]]
    let detail =
      if lastError[seat].len > 0: lastError[seat]
      else: "LLM decision failed"
    raisePlayerFailure(seat, "seat " & $seat & " rejected action: " & detail)

# ---------------------------------------------------------------------------
# The decision driver: the batch cadence, the wall-clock floor, the batch cap
# and the play deadline. Separated from the websocket server so the whole
# policy is testable against a fake clock with no sockets involved.
# ---------------------------------------------------------------------------

type
  DecisionDriver* = ref object
    client*: LlmClient
    selection*: PromptSelection
    prompts*: seq[string]
    scriptedKinds*: seq[ScriptKind]
    minBatchSeconds*: int
    maxBatches*: int
    batches*: int
    lastBatchAt*: float
    deadline*: float
    clock*: proc (): float {.closure.}
    sleeper*: proc (seconds: float) {.closure.}
    deadlineHit*: bool
    batchSizes*: seq[int]

proc newDecisionDriver*(client: LlmClient, config: GameConfig,
    prompts: seq[string], scriptedKinds: seq[ScriptKind], selection: PromptSelection,
    clock: proc (): float {.closure.} = nil,
    sleeper: proc (seconds: float) {.closure.} = nil): DecisionDriver =
  DecisionDriver(
    client: client,
    selection: selection,
    prompts: prompts,
    scriptedKinds: scriptedKinds,
    minBatchSeconds: config.minBatchSeconds,
    maxBatches: config.maxDecisionBatches,
    batches: 0,
    lastBatchAt: -1.0e9,
    deadline: playDeadlineSeconds(config),
    clock: clock,
    sleeper: sleeper
  )

proc elapsed(driver: DecisionDriver): float =
  if driver.clock == nil: 0.0 else: driver.clock()

proc pastDeadline*(driver: DecisionDriver): bool =
  driver.clock != nil and driver.elapsed() >= driver.deadline

proc decide*(driver: DecisionDriver, sim: var Sim, seats: seq[int],
    cause: string): seq[Decision] =
  ## One batch for every eligible seat at this decision point.
  if seats.len == 0:
    return @[]
  result = newSeq[Decision](seats.len)
  var hasModelSeat = false
  for seat in seats:
    if driver.scriptedKinds[seat] == skNone:
      hasModelSeat = true
      break
  if driver.batches >= driver.maxBatches or driver.pastDeadline():
    if driver.pastDeadline():
      driver.deadlineHit = true
    if hasModelSeat:
      if driver.pastDeadline():
        raiseInfraFailure("play deadline reached at decision point")
      raiseInfraFailure("decision batch limit (" & $driver.maxBatches &
        ") reached before episode completed")
    for index, seat in seats:
      result[index] = scriptedDecision(sim, seat, driver.scriptedKinds[seat],
        sim.inMeeting)
      result[index].source = dsScripted
      result[index].latencyMs = 0
    return
  ## `minBatchSeconds` floors the spacing between batch STARTS, so the episode
  ## stays under the Bedrock sidecar's 30 rpm per-episode ceiling (raid,
  ## 2026-08-23).
  if driver.minBatchSeconds > 0 and driver.clock != nil and
      driver.batches > 0:
    let wait = driver.minBatchSeconds.float -
      (driver.elapsed() - driver.lastBatchAt)
    if wait > 0.0 and driver.sleeper != nil:
      driver.sleeper(wait)
  driver.lastBatchAt = driver.elapsed()
  driver.batches.inc
  driver.batchSizes.add(seats.len)
  result = decideAll(driver.client, sim, seats, driver.prompts,
    driver.scriptedKinds, cause, driver.selection)
