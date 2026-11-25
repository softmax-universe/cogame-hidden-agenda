## tests/test_llm.nim — the decision layer.
##
## The transport itself is never called here: what is tested is the tolerant
## extraction, the strict reply schema, the batch shape, the wall-clock floor,
## the batch cap and the play deadline.

import std/[json, strutils, unicode]
import curly
import support/helpers
import hidden_agenda/[sim_types, sim_config, sim_state, sim, llm, prompts]

template check(condition: bool, message: string) =
  if not condition:
    echo "FAIL: ", message
    quit(1)

proc rig(variant = "hidden-agenda-notalk"): Sim =
  var config = baseConfig(4, variant)
  config.impostorSlot = 4
  result = initSim(config)

proc parses(sim: Sim, slot: int, text: string): bool =
  try:
    discard parseReply(sim, slot, extractJsonObject(text))
    true
  except CatchableError:
    false

block extraction:
  let bare = "{\"plan\":[{\"job\":\"guard\"}]}"
  check(extractJsonObject(bare){"plan"}.len == 1,
    "a bare JSON object is extracted")
  let fencedJson = "```json\n{\"plan\":[{\"job\":\"guard\"}]}\n```"
  check(extractJsonObject(fencedJson){"plan"}.len == 1,
    "a json-tagged outer markdown fence is extracted")
  let fencedPlain = "```\n{\"plan\":[{\"job\":\"guard\"}]}\n```"
  check(extractJsonObject(fencedPlain){"plan"}.len == 1,
    "a plain outer markdown fence is extracted")
  let bomAndBraces = "\xEF\xBB\xBF```json\n" &
    "{\"plan\":[{\"job\":\"guard\"}],\"notes\":\"watch \\\"}\\\" and {NW}\"}\n```"
  let extracted = extractJsonObject(bomAndBraces)
  check(extracted{"plan"}.len == 1,
    "BOM, single outer fence, and string braces parse cleanly")
  check(extracted{"notes"}.getStr() == "watch \"}\" and {NW}",
    "string-internal braces and escaped quotes survive parsing")

  for bad in [
    ("leading prose",
      "Sure! Here is my plan.\n{\"plan\":[{\"job\":\"hold\"}]}"),
    ("trailing prose",
      "{\"plan\":[{\"job\":\"hold\"}]}\nHope that helps."),
    ("prose around fence",
      "Here you go:\n```json\n{\"plan\":[{\"job\":\"hold\"}]}\n```"),
    ("trailing prose after fence",
      "```json\n{\"plan\":[{\"job\":\"hold\"}]}\n```\nExtra note mentioning {\"job\":\"guard\"}."),
    ("extra JSON object",
      "{\"plan\":[{\"job\":\"hold\"}]} {\"plan\":[{\"job\":\"guard\"}]}"),
    ("two fenced objects",
      "```json\n{\"plan\":[{\"job\":\"hold\"}]}\n```\n```json\n{\"plan\":[{\"job\":\"guard\"}]}\n```"),
    ("unsupported fence tag",
      "```python\n{\"plan\":[{\"job\":\"hold\"}]}\n```"),
    ("no object at all",
      "I would rather not answer that."),
    ("unbalanced JSON object",
      "{\"plan\":[{\"job\":\"hold\"}]"),
    ("non-object JSON array",
      "[{\"job\":\"hold\"}]")
  ]:
    var raised = false
    try:
      discard extractJsonObject(bad[1])
    except CatchableError:
      raised = true
    check(raised, bad[0] & " must be rejected")

block schema:
  var sim = rig()
  ## A crew reply containing an impostor-only job is INVALID.
  check(not parses(sim, 0, "{\"plan\":[{\"job\":\"hunt\",\"who\":\"PINK\"}]}"),
    "crew may not hunt")
  check(parses(sim, 4, "{\"plan\":[{\"job\":\"hunt\",\"who\":\"RED\"}]}"),
    "the impostor may")
  ## A plan of four steps is invalid.
  check(not parses(sim, 0, "{\"plan\":[{\"job\":\"hold\"},{\"job\":\"hold\"}," &
    "{\"job\":\"hold\"},{\"job\":\"hold\"}]}"), "four steps is too many")
  check(not parses(sim, 0, "{\"plan\":[]}"), "an empty plan is invalid")
  check(not parses(sim, 0, "{\"vote\":\"skip\"}"), "a missing plan is invalid")
  ## `mine` without `at`.
  check(not parses(sim, 0, "{\"plan\":[{\"job\":\"mine\"}]}"),
    "mine needs a seam")
  check(not parses(sim, 0, "{\"plan\":[{\"job\":\"mine\",\"at\":\"S9\"}]}"),
    "and a REAL seam")
  check(parses(sim, 0, "{\"plan\":[{\"job\":\"mine\",\"at\":\"s5\"}]}"),
    "seam ids are case-insensitive")
  ## `watch` naming a frozen cog.
  sim.cogs[2].state = csFrozen
  check(not parses(sim, 0,
    "{\"plan\":[{\"job\":\"watch\",\"who\":\"GREEN\"}]}"),
    "watch may not name a frozen cog")
  check(parses(sim, 0, "{\"plan\":[{\"job\":\"watch\",\"who\":\"PINK\"}]}"),
    "but may name an active one")
  ## Rooms.
  check(not parses(sim, 0,
    "{\"plan\":[{\"job\":\"patrol\",\"room\":\"MOON\"}]}"),
    "patrol needs a real room")
  check(parses(sim, 0, "{\"plan\":[{\"job\":\"patrol\",\"room\":\"NW\"}]}"),
    "and accepts one")

block meetingSchema:
  var sim = rig()
  sim.openMeetingForTest(mcCadence)
  sim.cogs[3].state = csEjected
  const plan = "\"plan\":[{\"job\":\"guard\"}]"
  check(not parses(sim, 0, "{" & plan & "}"),
    "a vote is required at a meeting")
  check(not parses(sim, 0, "{" & plan & ",\"vote\":\"YELLOW\"}"),
    "a vote for an ejected cog is invalid")
  check(parses(sim, 0, "{" & plan & ",\"vote\":\"skip\"}"), "skip is legal")
  check(parses(sim, 0, "{" & plan & ",\"vote\":\"PINK\"}"),
    "an active alias is legal")
  check(not parses(sim, 0, "{" & plan &
    ",\"vote\":\"PINK\",\"switch\":{\"if\":\"YELLOW\",\"to\":\"PINK\"}}"),
    "a switch naming an inactive cog is invalid")
  check(parses(sim, 0, "{" & plan &
    ",\"vote\":\"PINK\",\"switch\":{\"if\":\"tie\",\"to\":\"skip\"}}"),
    "if:tie / to:skip is legal")
  check(parses(sim, 0, "{" & plan & ",\"vote\":\"PINK\",\"switch\":null}"),
    "a null switch is no conditional at all")

block compactJobShimsAreRejectedAndPromptsAreCanonical:
  var sim = rig()
  ## Legacy compact job strings inside "job" are rejected; arguments must be
  ## in their sibling JSON keys.
  for legacy in [
    (0, "{\"plan\":[{\"job\":\"mine at:S2\"}]}"),
    (0, "{\"plan\":[{\"job\":\"watch who:PINK\"}]}"),
    (0, "{\"plan\":[{\"job\":\"patrol room:NW\"}]}"),
    (4, "{\"plan\":[{\"job\":\"hunt who:GREEN\"}]}"),
    (4, "{\"plan\":[{\"job\":\"strike who:RED\"}]}"),
    (4, "{\"plan\":[{\"job\":\"lurk room:SE\"}]}")
  ]:
    check(not parses(sim, legacy[0], legacy[1]),
      "compact job shim must be rejected: " & legacy[1])

  check(not parses(sim, 0, "{\"plan\":[{\"job\":\"teleport\"}]}"),
    "a genuinely unknown job is invalid")

  ## Both systemPrompt and userPrompt teach the canonical JSON step objects.
  let sysCrew = systemPrompt(sim, 0, PromptSelection())
  check("{\"job\":\"mine\",\"at\":\"<seam>\"}" in sysCrew,
    "systemPrompt teaches canonical JSON step shape")
  check("mine at:" notin sysCrew,
    "systemPrompt does not teach contradictory compact job syntax")
  let sysImp = systemPrompt(sim, 4, PromptSelection())
  check("{\"job\":\"hunt\",\"who\":\"<cog>\"}" in sysImp,
    "impostor systemPrompt teaches canonical JSON step shape")
  check("hunt who:" notin sysImp,
    "impostor systemPrompt does not teach contradictory compact job syntax")

  let hint = userPrompt(sim, 0, "", "opening")
  check("top-level keys are ONLY plan, hunch, notes" in hint,
    "opening prompt forbids top-level step keys")
  check("put \"job\", \"at\", \"who\", and \"room\" ONLY inside \"plan\" step objects" in hint,
    "prompt explicitly keeps step keys inside plan")
  var meetingNoTalk = rig("hidden-agenda-notalk")
  meetingNoTalk.openMeetingForTest(mcCadence)
  check("top-level keys are ONLY plan, vote, switch, hunch, notes" in
    userPrompt(meetingNoTalk, 0, "", "meeting"),
    "no-talk meeting prompt lists exact meeting top-level keys")
  var meetingTalk = rig("hidden-agenda")
  meetingTalk.openMeetingForTest(mcCadence)
  check("top-level keys are ONLY plan, vote, switch, say, hunch, notes" in
    userPrompt(meetingTalk, 0, "", "meeting"),
    "chat meeting prompt lists say among top-level keys")
  check("{\"job\":\"mine\",\"at\":\"S2\"}" in hint,
    "the schema hint spells mine's sibling key")
  check("{\"job\":\"patrol\",\"room\":\"NW\"}" in hint, "and patrol's")
  check("{\"job\":\"watch\",\"who\":\"BLUE\"}" in hint,
    "and watch's, naming a cog that is active now and is not this seat")
  check("{\"job\":\"lurk\",\"room\":\"SE\"}" in userPrompt(sim, 4, "",
    "opening"), "the impostor's jobs are spelled out too")
  check(not ("{\"job\":\"lurk\"" in hint),
    "and a crew seat is never shown an impostor-only job")
  check(parses(sim, 0, "{\"plan\":[{\"job\":\"watch\",\"who\":\"BLUE\"}]}"),
    "every step the hint shows is itself a valid step")

block aHalfWrittenSwitchIsRejected:
  var sim = rig()
  sim.openMeetingForTest(mcCadence)
  const plan = "\"plan\":[{\"job\":\"guard\"}]"
  for half in ["{\"if\":\"tie\"}", "{\"to\":\"skip\"}", "{}"]:
    let reply = "{" & plan & ",\"vote\":\"PINK\",\"switch\":" & half & "}"
    check(not parses(sim, 0, reply),
      "a half-written switch " & half & " must be rejected")
  ## Omitted switch and null switch remain valid optional defaults.
  check(parses(sim, 0, "{" & plan & ",\"vote\":\"PINK\"}"),
    "omitted switch is valid")
  check(parses(sim, 0, "{" & plan & ",\"vote\":\"PINK\",\"switch\":null}"),
    "null switch is valid")

block legacyScriptKindAliasesAreRejected:
  check(parseScriptKind("miner") == skMiner, "miner is valid")
  check(parseScriptKind("lurker") == skLurker, "lurker is valid")
  for legacy in ["1", "true", "yes", "lurk", "bot"]:
    check(parseScriptKind(legacy) == skNone,
      "legacy alias '" & legacy & "' must not resolve to a baseline")

block sayIsIgnoredNotRejected:
  var sim = rig("hidden-agenda-notalk")
  sim.openMeetingForTest(mcCadence)
  let reply = "{\"plan\":[{\"job\":\"guard\"}],\"vote\":\"skip\"," &
    "\"say\":\"i saw pink do it\"}"
  check(parses(sim, 0, reply),
    "a say in a no-talk variant is IGNORED, not an error")
  let decision = parseReply(sim, 0, extractJsonObject(reply))
  check(decision.say.len == 0, "and is never recorded")
  var chat = rig("hidden-agenda")
  chat.openMeetingForTest(mcCadence)
  let kept = parseReply(chat, 0, extractJsonObject(reply))
  check(kept.say == "i saw pink do it", "the chat variant keeps it")

block capsAreRuneSafe:
  var sim = rig("hidden-agenda")
  sim.openMeetingForTest(mcCadence)
  var long = "{\"plan\":[{\"job\":\"guard\"}],\"vote\":\"skip\",\"say\":\""
  for _ in 0 ..< 300:
    long.add("\\u00e9")
  long.add("\",\"hunch\":\"")
  for _ in 0 ..< 300:
    long.add("\\u4e2d")
  long.add("\",\"notes\":\"")
  for _ in 0 ..< 400:
    long.add("\\u2026")
  long.add("\"}")
  let decision = parseReply(sim, 0, extractJsonObject(long))
  check(decision.say.runeLen <= MaxSayLen, "say is capped in runes")
  check(decision.hunch.runeLen <= MaxHunchLen, "hunch is capped in runes")
  check(decision.notes.runeLen <= MaxNotesLen, "notes is capped in runes")
  check(validateUtf8(decision.notes) == -1, "and the cut is never mid-rune")

proc okStubClient(): LlmClient =
  const good = """{"plan":[{"job":"guard"}],"vote":"skip"}"""
  proc send(batch: RequestBatch, timeoutSeconds: int):
      ResponseBatch {.closure.} =
    for _ in 0 ..< batch.len:
      result.add((Response(code: 200, body: $(%*{
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": good}]
      })), ""))
  stubbedLlmClient(send)

block oneBatchCarriesEveryEligibleSeat:
  var sim = rig()
  let client = okStubClient()
  var prompts: seq[string]
  var kinds: seq[ScriptKind]
  for _ in 0 ..< Seats:
    prompts.add("play well")
    kinds.add(skNone)
  var now = 0.0
  var slept = 0.0
  proc clock(): float {.closure.} = now
  proc sleeper(seconds: float) {.closure.} =
    slept += seconds
    now += seconds
  let driver = newDecisionDriver(client, sim.config, prompts, kinds, PromptSelection(), clock,
    sleeper)
  var seats = sim.activeSeats()
  check(seats.len == Seats, "five seats at the opening")
  var decisions = driver.decide(sim, seats, "opening")
  check(decisions.len == Seats, "one decision per seat")
  for decision in decisions:
    check(decision.source == dsLlm,
      "valid LLM replies are recorded as llm")
    check(decision.plan.len >= 1, "and carry a legal plan")
  sim.cogs[2].state = csFrozen
  seats = sim.activeSeats()
  check(seats.len == Seats - 1, "a frozen seat is not in the batch")
  decisions = driver.decide(sim, seats, "meeting")
  check(decisions.len == Seats - 1, "the batch shrinks with the roster")
  check(driver.batchSizes == @[Seats, Seats - 1],
    "exactly two batches were issued, of 5 then 4 requests")

block disabledClientFailsPromptSeatsButAllowsScriptedSeats:
  var sim = rig()
  let client = disabledLlmClient()
  var prompts: seq[string]
  var scriptedKinds: seq[ScriptKind]
  var llmKinds: seq[ScriptKind]
  for _ in 0 ..< Seats:
    prompts.add("play well")
    scriptedKinds.add(skMiner)
    llmKinds.add(skNone)
  let seats = sim.activeSeats()
  let scripted = client.decideAll(sim, seats, prompts, scriptedKinds, "opening",
    PromptSelection())
  for decision in scripted:
    check(decision.source == dsScripted,
      "genuine scripted seats still run without LLM credentials")
  var infraRaised = false
  try:
    discard client.decideAll(sim, seats, prompts, llmKinds, "opening",
      PromptSelection())
  except InfraFailureError:
    infraRaised = true
  check(infraRaised,
    "prompt seat without LLM credentials raises InfraFailureError instead of blaming a player or falling back")

block minBatchSecondsFloorsTheSpacing:
  var sim = rig()
  var config = sim.config
  config.minBatchSeconds = 14
  let client = okStubClient()
  var prompts: seq[string]
  var kinds: seq[ScriptKind]
  for _ in 0 ..< Seats:
    prompts.add("play well")
    kinds.add(skNone)
  var now = 0.0
  proc clock(): float {.closure.} = now
  proc sleeper(seconds: float) {.closure.} = now += seconds
  let driver = newDecisionDriver(client, config, prompts, kinds, PromptSelection(), clock, sleeper)
  let seats = sim.activeSeats()
  discard driver.decide(sim, seats, "opening")
  let first = now
  discard driver.decide(sim, seats, "meeting")
  check(now - first >= 14.0,
    "batch STARTS are floored " & $config.minBatchSeconds & " s apart, got " &
    $(now - first))

block maxDecisionBatchesCaps:
  var sim = rig()
  var config = sim.config
  config.minBatchSeconds = 0
  config.maxDecisionBatches = 3
  let client = okStubClient()
  var prompts: seq[string]
  var kinds: seq[ScriptKind]
  for _ in 0 ..< Seats:
    prompts.add("play well")
    kinds.add(skNone)
  proc clock(): float {.closure.} = 0.0
  proc sleeper(seconds: float) {.closure.} = discard
  let driver = newDecisionDriver(client, config, prompts, kinds, PromptSelection(), clock, sleeper)
  let seats = sim.activeSeats()
  for round in 0 ..< 3:
    discard driver.decide(sim, seats, "meeting")
  check(driver.batchSizes.len == 3, "three batches were issued")
  var infraRaised = false
  try:
    discard driver.decide(sim, seats, "meeting")
  except InfraFailureError:
    infraRaised = true
  check(driver.batchSizes.len == 3, "and never a fourth")
  check(infraRaised,
    "past the cap non-scripted seats raise InfraFailureError instead of reusing stale previous moves")

block theDeadlineSettlesEarly:
  var config = baseConfig(4)
  config.impostorSlot = 4
  var sim = initSim(config)
  var now = 0.0
  proc clock(): float {.closure.} = now
  proc decide(view: var Sim, seats: seq[int], cause: string):
      seq[Decision] {.closure.} =
    var decisions: seq[Decision]
    now += 30.0
    for seat in seats:
      decisions.add(scriptedDecision(view, seat, skMiner, view.inMeeting))
    decisions
  sim.runEpisode(decide, clock, 10.0)
  sim.finalise()
  check(sim.reason == "deadline", "the play deadline settles the episode")
  check(sim.ending == "deadline", "with the deadline ending")
  for score in sim.scores:
    check(score == 0, "and all scores 0")

block frozenAndEjectedSeatsAreToldTheyCannotAct:
  ## docs/POLICIES.md and the manifest's policies page both promise that a
  ## frozen or ejected seat receives a frame with `canAct: false`; seatView
  ## emitted no such key and left the seat to infer it from `you.state`.
  var sim = rig()
  var frame = seatView(sim, 0, "opening")
  check(frame{"canAct"}.getBool() == true,
    "an active seat in play can act")
  check(frame{"canVote"}.getBool() == false,
    "and cannot vote outside a meeting")

  sim.openMeetingForTest(mcCadence)
  frame = seatView(sim, 0, "meeting")
  check(frame{"canAct"}.getBool() and frame{"canVote"}.getBool(),
    "an active seat in a meeting can do both")

  sim.cogs[1].state = csFrozen
  sim.cogs[2].state = csEjected
  for slot in [1, 2]:
    let dead = seatView(sim, slot, "meeting")
    check(dead{"canAct"}.getBool() == false,
      "a " & $sim.cogs[slot].state & " seat is told canAct: false")
    check(dead{"canVote"}.getBool() == false, "and canVote: false")
    check(dead{"you"}{"state"}.getStr() in ["csFrozen", "csEjected",
      "frozen", "ejected"], "and still carries you.state")
  ## And that is the truth: neither seat is in the batch.
  let seats = sim.activeSeats()
  check(1 notin seats and 2 notin seats,
    "the flags agree with activeSeats()")

block singleAttemptFailureAttributionIsDriven:
  ## `stubbedLlmClient` installs a batch sender in place of curl so we verify
  ## single-attempt (no retry) behavior and exact failing-seat attribution for
  ## both provider/transport errors and malformed model replies.

  proc reply(text: string): Response =
    Response(code: 200, body: $(%*{
      "stop_reason": "end_turn",
      "content": [{"type": "text", "text": text}]
    }))

  let good = """{"plan":[{"job":"guard"}],"vote":"skip"}"""

  ## (1) A transport error on seat 2 fails immediately with one batch and
  ## attributes InfraFailureError to seat 2 (never retried, never seat 0).
  block:
    var sim = rig()
    var batches = 0
    proc send(batch: RequestBatch, timeoutSeconds: int):
        ResponseBatch {.closure.} =
      batches.inc
      for i in 0 ..< batch.len:
        if batch[i].tag == "2":
          result.add((Response(), "connection timed out"))
        else:
          result.add((reply(good), ""))
    let client = stubbedLlmClient(send)
    var prompts: seq[string]
    var kinds: seq[ScriptKind]
    for _ in 0 ..< Seats:
      prompts.add("play well")
      kinds.add(skNone)
    let seats = sim.activeSeats()
    var failedSlot = -1
    try:
      discard client.decideAll(sim, seats, prompts, kinds, "opening", PromptSelection())
    except InfraFailureError as err:
      failedSlot = err.slot
    check(batches == 1, "a transport error must not retry, got " & $batches)
    check(failedSlot == 2,
      "transport error attributes InfraFailureError to actual seat 2, got " &
      $failedSlot)

  ## (2) A 429 fails immediately in one batch with InfraFailureError on the actual seat.
  block:
    var sim = rig()
    var batches = 0
    proc send(batch: RequestBatch, timeoutSeconds: int):
        ResponseBatch {.closure.} =
      batches.inc
      for i in 0 ..< batch.len:
        result.add((Response(code: 429, body: "slow down"), ""))
    let client = stubbedLlmClient(send)
    var prompts: seq[string]
    var kinds: seq[ScriptKind]
    for _ in 0 ..< Seats:
      prompts.add("play well")
      kinds.add(skNone)
    let seats = sim.activeSeats()
    var failedSlot = -1
    try:
      discard client.decideAll(sim, seats, prompts, kinds, "opening", PromptSelection())
    except InfraFailureError as err:
      failedSlot = err.slot
    check(batches == 1, "a 429 must not retry, got " & $batches & " batches")
    check(failedSlot == 0,
      "429 raises InfraFailureError for seat 0, got " & $failedSlot)

  ## (3) A 403 disables the client outright: one batch, raises InfraFailureError,
  ## and every LATER decision point also fails immediately without issuing a batch.
  block:
    var sim = rig()
    var batches = 0
    proc send(batch: RequestBatch, timeoutSeconds: int):
        ResponseBatch {.closure.} =
      batches.inc
      for i in 0 ..< batch.len:
        result.add((Response(code: 403, body: "no"), ""))
    let client = stubbedLlmClient(send)
    var prompts: seq[string]
    var kinds: seq[ScriptKind]
    for _ in 0 ..< Seats:
      prompts.add("play well")
      kinds.add(skNone)
    let seats = sim.activeSeats()
    var failedSlot = -1
    try:
      discard client.decideAll(sim, seats, prompts, kinds, "opening", PromptSelection())
    except InfraFailureError as err:
      failedSlot = err.slot
    check(batches == 1, "a 403 stops inside the first batch")
    check(client.disabled, "and disables the client")
    check(failedSlot == 0, "and raises InfraFailureError on seat 0")
    failedSlot = -1
    try:
      discard client.decideAll(sim, seats, prompts, kinds, "meeting", PromptSelection())
    except InfraFailureError as err:
      failedSlot = err.slot
    check(batches == 1, "a disabled client never issues another batch")
    check(failedSlot == 0, "and raises InfraFailureError on later calls")

  ## (4) Junk on seat 3: one batch (no retry), then PlayerFailureError on seat 3.
  block:
    var sim = rig()
    var batches = 0
    proc send(batch: RequestBatch, timeoutSeconds: int):
        ResponseBatch {.closure.} =
      batches.inc
      for i in 0 ..< batch.len:
        if batch[i].tag == "3":
          result.add((reply("I would rather not say."), ""))
        else:
          result.add((reply(good), ""))
    let client = stubbedLlmClient(send)
    var prompts: seq[string]
    var kinds: seq[ScriptKind]
    for _ in 0 ..< Seats:
      prompts.add("play well")
      kinds.add(skNone)
    let seats = sim.activeSeats()
    var failedSlot = -1
    try:
      discard client.decideAll(sim, seats, prompts, kinds, "opening", PromptSelection())
    except PlayerFailureError as err:
      failedSlot = err.slot
    check(batches == 1, "junk must not be retried, got " & $batches)
    check(failedSlot == 3,
      "junk raises PlayerFailureError on actual failing seat 3, got " &
      $failedSlot)


echo "test_llm: ok"
