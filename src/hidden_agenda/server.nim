## The Hidden Agenda game server: the Coworld game contract over mummy.
##
## Fork of `coworld-ctf/src/ctf/server.nim`'s route / artifact / shutdown
## skeleton; the player protocol becomes bullwhip's JSON frames.
##
## Routes — hosted certification probes exactly these BEFORE the player pods
## start (lantern, 2026-08-23), so all of them serve real pages:
##   GET /healthz                     200 ok, until shutdownGraceSeconds after
##                                    the artifacts are written
##   GET /client/player?slot=&token=  the seat's HTML shell; it NEVER opens the
##                                    player socket
##   GET /client/global               the broadcast client
##
## There is deliberately NO pod route serving a replay page. A finished episode
## is watched through the STATIC wasm bundle the platform builds from
## tools/build_replay_viewer.sh (coworld_manifest_template.json declares
## "replay_viewer": {"bundle": "static-replay-viewer"}), which contacts nothing
## but S3. A pod-served replay page would keep a game container alive just to
## watch a finished episode, and tests/test_manifest.nim asserts the path is
## absent from src/, docs/, client/, the README and the manifest.
##   WS  /player?slot=N&token=T       the seat socket; a bad token is refused
##                                    with a close, never a hang
##   WS  /global                      live spectator: the packet + chrome frame
##
## Prompt decisions use the game-hosted Claude adapter. External policies
## receive their private observation and return a normal action. The game
## validates actions and owns rules, visibility, results, and replay.

import std/[json, locks, os, sets, strutils, tables, times, unicode]
import bitworld/runtime
import curly
import mummy
import mummy/routers
import sim_types, sim_config, sim_state, sim, scripted, llm,
  global, replays, wire_constants, prompts

const
  PlayerPage = staticRead("../../client/player.html")
  GlobalPage = staticRead("../../client/global.html")
  ChromeCommonJs = staticRead("../../client/chrome_common.js")
  BroadcastCoreJs = staticRead("../../client/broadcast_core.js")
  ChromeCommonMarker = "<!-- CHROME_COMMON -->"
  BroadcastCoreMarker = "<!-- BROADCAST_CORE -->"

type
  ServerState = object
    prompts: seq[string]
    scriptedKinds: seq[ScriptKind]
    external: seq[bool]
    decisionId: int
    pendingActions: seq[JsonNode]
    pendingErrors: seq[string]
    registered: seq[bool]
    everRegistered: seq[bool]
    disconnected: seq[bool]
    playerSockets: Table[int, WebSocket]
    socketSlots: Table[WebSocket, int]
    globalSockets: HashSet[WebSocket]
    snapshot: string
    seats: int
    finished: bool

var
  stateLock: Lock
  shared: ServerState
  gameSim: Sim
  gameServer: Server
  gameConfig: GameConfig

initLock(stateLock)

proc splicePage(page: string): string =
  result = spliceWireConstants(page)
  result = result.replace(ChromeCommonMarker,
    "<script>" & ChromeCommonJs & "</script>")
  result = result.replace(BroadcastCoreMarker,
    "<script>" & BroadcastCoreJs & "</script>")

proc refreshSnapshotLocked() =
  shared.snapshot = globalSnapshot(gameSim)
  for socket in shared.globalSockets:
    try:
      socket.send(shared.snapshot)
    except CatchableError:
      discard

proc declarePlayerFailure(slot: int, message: string) =
  try:
    writeCogameEnv("COGAME_PLAYER_FAILURE_URI",
      $(%*{"failed_policy_index": slot, "message": message}),
      "application/json")
  except CatchableError as error:
    echo "hidden-agenda: player-failure declaration failed: ", error.msg

proc writeArtifact(uri, data, contentType, methodEnv: string) =
  if uri.len == 0:
    return
  let httpMethod = getEnv(methodEnv, "PUT").toUpperAscii()
  if uri.isHttpCogameUri() and httpMethod == "POST":
    let curl = newCurly()
    var headers: HttpHeaders
    headers["content-type"] = contentType
    let response = curl.post(uri, headers, data, 60)
    if response.code < 200 or response.code >= 300:
      raise newException(IOError, "artifact POST failed: " & $response.code)
  else:
    writeCogameUri(uri, data, contentType, methodEnv)

proc welcomeFrame(slot: int): string =
  var aliases = newJArray()
  for alias in Aliases:
    aliases.add(%alias)
  ## `role` is THIS SEAT'S OWN role and nothing else's.
  $ %*{
    "type": "welcome", "protocol": Protocol, "slot": slot,
    "role": $gameSim.cogs[slot].role, "name": Aliases[slot],
    "variant": gameConfig.variant, "chat": gameConfig.chat,
    "maxTicks": gameConfig.maxTicks,
    "depositTarget": gameConfig.depositTarget,
    "aliases": aliases
  }

proc pushStateFrames(cause: string) =
  for slot, socket in shared.playerSockets:
    if slot < 0 or slot >= Seats:
      continue
    try:
      socket.send($seatView(gameSim, slot, cause))
    except CatchableError:
      discard

proc finalFrame(slot: int): string =
  let results = gameSim.resultsJson()
  var names = newJArray()
  for alias in Aliases:
    names.add(%alias)
  ## `slot` is THIS seat's own slot, the same field the `welcome` and `state`
  ## frames carry, so a policy that multiplexes seats can tell whose buzzer
  ## this is. NO roles[], NO impostorSlot, NO seed: nobody learns the answer
  ## from the game, not even at the buzzer.
  $ %*{
    "type": "final", "done": true, "slot": slot,
    "scores": results{"scores"}, "win": results{"win"},
    "winner": gameSim.winner, "names": names,
    "deposits": gameSim.deposits, "ticks": gameSim.frames.len,
    "reason": gameSim.reason, "ending": gameSim.ending
  }

proc broadcastFinal() =
  var allowance = epochTime()
  for slot, socket in shared.playerSockets:
    allowance += 3.0
    if epochTime() > allowance:
      echo "hidden-agenda: final broadcast past budget; skipping slot ", slot
      continue
    try:
      socket.send(finalFrame(slot))
    except CatchableError as error:
      echo "hidden-agenda: final frame to slot ", slot, " failed: ", error.msg

proc finishEpisode(runtimeConfig: RuntimeConfig) =
  var results: JsonNode
  var replayData: string
  withLock stateLock:
    if shared.finished:
      return
    shared.finished = true
    gameSim.finalise()
    results = gameSim.resultsJson()
    replayData = replayBytes(gameSim)
    ## Final frames to the players BEFORE the artifacts: the hosted worker
    ## tears player pods down as soon as results.json exists.
    broadcastFinal()
    refreshSnapshotLocked()
  sleep(500)
  echo "hidden-agenda: writing results and replay (", replayData.len,
    " bytes)"
  writeArtifact(runtimeConfig.resultsUri, $results, "application/json",
    "COGAME_RESULTS_METHOD")
  writeArtifact(runtimeConfig.replayUri, replayData, "application/json",
    "COGAME_SAVE_REPLAY_METHOD")
  echo "hidden-agenda: episode complete (", gameSim.reason, "/",
    gameSim.ending, ", winner ", gameSim.winner, ") after ",
    gameSim.frames.len, " ticks, ", gameSim.deposits, " deposits, ",
    gameSim.freezes, " freezes (", gameSim.witnessedFreezes, " witnessed), ",
    gameSim.meetings.len, " meetings"

proc runGame(runtimeConfig: RuntimeConfig) {.gcsafe.} =
  {.gcsafe.}:
    let config = gameConfig
    let selection = promptSelection(parseJson(runtimeConfig.config))
    let gameStart = epochTime()
    let connectDeadline = gameStart +
      config.playerConnectTimeoutSeconds.float
    while epochTime() < connectDeadline:
      var allConnected = false
      withLock stateLock:
        allConnected = shared.playerSockets.len >= shared.seats
      if allConnected:
        break
      sleep(200)
    ## Give a connected-but-silent seat a moment to send its prompt frame.
    let registerDeadline = min(epochTime() + 3.0, connectDeadline + 3.0)
    while epochTime() < registerDeadline:
      var allRegistered = true
      withLock stateLock:
        for slot in 0 ..< shared.seats:
          if shared.playerSockets.hasKey(slot) and not shared.registered[slot]:
            allRegistered = false
      if allRegistered:
        break
      sleep(100)

    var connected = 0
    var noShow = -1
    withLock stateLock:
      connected = shared.playerSockets.len
      for slot in 0 ..< shared.seats:
        if not shared.everRegistered[slot]:
          if noShow < 0:
            noShow = slot
      echo "hidden-agenda: starting with ", connected, "/", shared.seats,
        " players connected"
      refreshSnapshotLocked()
    if noShow >= 0:
      let msg =
        if connected == 0:
          "no player connected within connect timeout"
        else:
          "player slot " & $noShow & " never registered"
      echo "hidden-agenda: ", msg, "; forfeiting"
      declarePlayerFailure(noShow, msg)
      gameSim.settle("forfeit", "forfeit", "none")
      finishEpisode(runtimeConfig)
      sleep(config.shutdownGraceSeconds * 1000)
      quit(0)

    var prompts: seq[string]
    var kinds: seq[ScriptKind]
    withLock stateLock:
      prompts = shared.prompts
      kinds = shared.scriptedKinds
    let client = newLlmClient(config)
    proc clock(): float {.closure.} = epochTime() - gameStart
    proc sleeper(seconds: float) {.closure.} =
      sleep(int(seconds * 1000.0))
    let driver = newDecisionDriver(client, config, prompts, kinds, selection,
      clock, sleeper)

    proc decide(view: var Sim, seats: seq[int], cause: string):
        seq[Decision] {.closure.} =
      result = newSeq[Decision](seats.len)
      var external: seq[bool]
      var internalSeats: seq[int]
      var internalIndices: seq[int]
      var externalSeats: seq[int]
      var externalIndices: seq[int]
      var decisionId: int
      var requestExternal = false
      var disconnectedSeat = -1
      withLock stateLock:
        pushStateFrames(cause)
        driver.prompts = shared.prompts
        driver.scriptedKinds = shared.scriptedKinds
        external = shared.external
        for index, seat in seats:
          if shared.disconnected[seat] and
              (external[seat] or shared.scriptedKinds[seat] == skNone):
            disconnectedSeat = seat
            break
          if external[seat]:
            externalSeats.add(seat)
            externalIndices.add(index)
          else:
            internalSeats.add(seat)
            internalIndices.add(index)
        requestExternal = externalSeats.len > 0 and
          driver.batches < driver.maxBatches and not driver.pastDeadline()
        shared.pendingActions = newSeq[JsonNode](shared.seats)
        shared.pendingErrors = newSeq[string](shared.seats)
        if disconnectedSeat < 0 and requestExternal:
          inc shared.decisionId
          decisionId = shared.decisionId
          for seat in externalSeats:
            if shared.playerSockets.hasKey(seat):
              try:
                shared.playerSockets[seat].send($ %*{
                  "type": "observation", "id": decisionId,
                  "observation": seatView(view, seat, cause)})
              except CatchableError:
                shared.disconnected[seat] = true
      if disconnectedSeat >= 0:
        raisePlayerFailure(disconnectedSeat,
          "player slot " & $disconnectedSeat & " disconnected")
      if internalSeats.len > 0:
        let internalDecisions = driver.decide(view, internalSeats, cause)
        for i, index in internalIndices:
          result[index] = internalDecisions[i]
      elif requestExternal:
        driver.batches.inc
        driver.batchSizes.add(externalSeats.len)
      if externalSeats.len > 0:
        if requestExternal:
          let started = epochTime()
          let deadline = epochTime() + config.llmTimeoutSeconds.float
          while epochTime() < deadline:
            var ready = true
            withLock stateLock:
              for seat in externalSeats:
                if shared.pendingActions[seat].isNil and
                    shared.pendingErrors[seat].len == 0 and
                    not shared.disconnected[seat] and
                    shared.playerSockets.hasKey(seat):
                  ready = false
            if ready:
              break
            sleep(20)
          for i, seat in externalSeats:
            let index = externalIndices[i]
            var action: JsonNode
            var errMsg: string
            var lost = false
            withLock stateLock:
              action = shared.pendingActions[seat]
              errMsg = shared.pendingErrors[seat]
              lost = shared.disconnected[seat] or
                not shared.playerSockets.hasKey(seat)
            if errMsg.len > 0:
              raisePlayerFailure(seat,
                "external player slot " & $seat & " rejected action: " & errMsg)
            if action.isNil:
              if lost:
                raisePlayerFailure(seat,
                  "external player slot " & $seat & " disconnected before acting")
              raisePlayerFailure(seat,
                "external player slot " & $seat & " timed out without submitting an action")
            try:
              result[index] = view.parseReply(seat, action)
            except CatchableError as error:
              raisePlayerFailure(seat,
                "external player slot " & $seat & " rejected action: " &
                cleanText(error.msg, MaxErrorLen))
            result[index].source = dsExternal
            result[index].latencyMs = int((epochTime() - started) * 1000)
            driver.previous[seat] = result[index]
            driver.hasPrevious[seat] = true
        else:
          if driver.pastDeadline():
            driver.deadlineHit = true
          for i, seat in externalSeats:
            let index = externalIndices[i]
            if driver.hasPrevious[seat]:
              result[index] = driver.previous[seat]
              result[index].source = dsBudget
              result[index].latencyMs = 0
            else:
              raisePlayerFailure(seat,
                "decision budget exhausted with no previous decision for seat " &
                $seat)

    proc onTick(view: var Sim) {.closure.} =
      if view.tick mod 24 == 0 or view.done:
        withLock stateLock:
          refreshSnapshotLocked()

    try:
      gameSim.runEpisode(decide, clock, playDeadlineSeconds(config), onTick)
    except PlayerFailureError as error:
      echo "hidden-agenda: player slot ", error.slot, " failed: ", error.msg
      declarePlayerFailure(error.slot, error.msg)
      gameSim.settle("forfeit", "forfeit", "none")
    withLock stateLock:
      pushStateFrames("final")
    finishEpisode(runtimeConfig)
    ## As cogame-lantern taught, /healthz and /global keep answering for
    ## shutdownGraceSeconds before quit(0), because hosted certification pings
    ## the global websocket AFTER the player pods start.
    echo "hidden-agenda: holding /healthz and /global for ",
      config.shutdownGraceSeconds, "s"
    sleep(config.shutdownGraceSeconds * 1000)
    quit(0)

var gameThread: Thread[RuntimeConfig]

proc healthzHandler(request: Request) {.gcsafe.} =
  var headers: HttpHeaders
  headers["Content-Type"] = "application/json"
  request.respond(200, headers, """{"ok": true}""")

proc respondHtml(request: Request, body: string) =
  var headers: HttpHeaders
  headers["Content-Type"] = "text/html; charset=utf-8"
  request.respond(200, headers, body)

proc playerPageHandler(request: Request) {.gcsafe.} =
  {.gcsafe.}:
    respondHtml(request, splicePage(PlayerPage))

proc globalPageHandler(request: Request) {.gcsafe.} =
  {.gcsafe.}:
    respondHtml(request, splicePage(GlobalPage))

proc clientAssetHandler(request: Request) {.gcsafe.} =
  {.gcsafe.}:
    let name = request.pathParams["name"]
    var headers: HttpHeaders
    headers["Content-Type"] = "application/javascript; charset=utf-8"
    case name
    of "chrome_common.js": request.respond(200, headers, ChromeCommonJs)
    of "broadcast_core.js": request.respond(200, headers, BroadcastCoreJs)
    of "wire_constants.js": request.respond(200, headers, WireConstantsJs)
    else: request.respond(404)

proc playerUpgradeHandler(request: Request) {.gcsafe.} =
  {.gcsafe.}:
    let slotText = request.queryParams["slot"]
    let token = request.queryParams["token"]
    var slot = -1
    try:
      slot = parseInt(slotText)
    except ValueError:
      discard
    var authorized = false
    var duplicate = false
    withLock stateLock:
      authorized = slot >= 0 and slot < gameConfig.tokens.len and
        gameConfig.tokens[slot] == token
      duplicate = authorized and shared.playerSockets.hasKey(slot)
    if not authorized:
      request.respond(403)
      return
    if duplicate:
      request.respond(409)
      return
    let websocket = request.upgradeToWebSocket()
    withLock stateLock:
      shared.playerSockets[slot] = websocket
      shared.socketSlots[websocket] = slot
      echo "hidden-agenda: player slot ", slot, " connected (",
        shared.playerSockets.len, "/", shared.seats, ")"
      try:
        websocket.send(welcomeFrame(slot))
      except CatchableError:
        discard

proc globalUpgradeHandler(request: Request) {.gcsafe.} =
  {.gcsafe.}:
    let websocket = request.upgradeToWebSocket()
    withLock stateLock:
      shared.globalSockets.incl(websocket)
      if shared.snapshot.len > 0:
        try:
          websocket.send(shared.snapshot)
        except CatchableError:
          discard

proc websocketHandler(websocket: WebSocket, event: WebSocketEvent,
    message: Message) {.gcsafe.} =
  {.gcsafe.}:
    case event
    of OpenEvent:
      discard
    of MessageEvent:
      ## mummy hands Ping frames to the application; the certifier pings
      ## /global to check the game is alive, so an unanswered ping fails
      ## certification.
      if message.kind == Ping:
        websocket.send(message.data, Pong)
        return
      if message.kind != TextMessage:
        return
      var slot = -1
      withLock stateLock:
        slot = shared.socketSlots.getOrDefault(websocket, -1)
      if slot < 0:
        return
      try:
        let payload = parseJson(message.data)
        if payload{"type"}.getStr() == "register":
          let control = payload{"control"}
          if control == nil or control.kind != JString or
              control.getStr() != "external":
            raise newException(HiddenAgendaError, "unknown player control")
          withLock stateLock:
            shared.external[slot] = true
            shared.registered[slot] = true
            shared.everRegistered[slot] = true
          return
        if payload{"type"}.getStr() == "action":
          withLock stateLock:
            let idNode = payload{"id"}
            if shared.external[slot] and idNode != nil and
                idNode.kind == JInt and idNode.getInt() == shared.decisionId and
                shared.pendingActions[slot].isNil and
                shared.pendingErrors[slot].len == 0:
              let action = payload{"action"}
              try:
                discard gameSim.parseReply(slot, action)
                shared.pendingActions[slot] = action
              except CatchableError as error:
                shared.pendingErrors[slot] = cleanText(error.msg, MaxErrorLen)
          return
        if payload{"type"}.getStr() != "prompt":
          echo "hidden-agenda: ignoring player frame of type '",
            payload{"type"}.getStr(), "'"
          return
        let promptNode = payload{"prompt"}
        if promptNode != nil and promptNode.kind notin {JNull, JString}:
          raise newException(HiddenAgendaError, "prompt must be a string or null")
        var prompt = if promptNode != nil and promptNode.kind == JString:
          promptNode.getStr() else: ""
        if prompt.runeLen > MaxPromptLen:
          prompt = cleanText(prompt, MaxPromptLen)
        let node = payload{"scripted"}
        var kind = skNone
        if node != nil and node.kind != JNull:
          if node.kind != JString:
            raise newException(HiddenAgendaError,
              "scripted must be a string or null")
          let scriptedText = node.getStr().strip()
          if scriptedText.len > 0:
            kind = parseScriptKind(scriptedText)
            if kind == skNone:
              raise newException(HiddenAgendaError,
                "unknown scripted baseline: " & scriptedText)
        if prompt.strip().len == 0 and kind == skNone:
          raise newException(HiddenAgendaError,
            "prompt frame must specify a prompt or scripted baseline")
        withLock stateLock:
          shared.prompts[slot] = prompt
          shared.scriptedKinds[slot] = kind
          shared.external[slot] = false
          shared.registered[slot] = true
          shared.everRegistered[slot] = true
        echo "hidden-agenda: slot ", slot, " registered (", prompt.len,
          " prompt chars", (if kind != skNone: ", scripted " & $kind
            else: ", llm"), ")"
      except CatchableError as error:
        echo "hidden-agenda: ignoring bad player frame: ", error.msg
    of ErrorEvent:
      discard
    of CloseEvent:
      withLock stateLock:
        if websocket in shared.socketSlots:
          let slot = shared.socketSlots[websocket]
          shared.socketSlots.del(websocket)
          if shared.playerSockets.getOrDefault(slot) == websocket:
            shared.playerSockets.del(slot)
          if not shared.finished:
            shared.disconnected[slot] = true
        shared.globalSockets.excl(websocket)

proc buildRouter(): Router =
  ## The two /client pages are registered BEFORE the asset route, so neither
  ## is shadowed by it. There is no replay page route: the replay viewer is
  ## the static bundle, never a pod path.
  result.get("/healthz", healthzHandler)
  result.get("/client/player", playerPageHandler)
  result.get("/client/global", globalPageHandler)
  result.get("/client/@name", clientAssetHandler)
  result.get("/global", globalUpgradeHandler)
  result.get("/player", playerUpgradeHandler)

proc stopServer*() =
  if gameServer != nil:
    gameServer.close()

proc runGameServer*(config: GameConfig, runtimeConfig: RuntimeConfig) =
  if config.tokens.len != config.numAgents:
    raise newException(HiddenAgendaError,
      "tokens must name exactly num_agents seats")
  gameConfig = config
  gameSim = initSim(config)
  shared.seats = config.numAgents
  shared.prompts = newSeq[string](shared.seats)
  shared.scriptedKinds = newSeq[ScriptKind](shared.seats)
  shared.external = newSeq[bool](shared.seats)
  shared.pendingActions = newSeq[JsonNode](shared.seats)
  shared.pendingErrors = newSeq[string](shared.seats)
  shared.registered = newSeq[bool](shared.seats)
  shared.everRegistered = newSeq[bool](shared.seats)
  shared.disconnected = newSeq[bool](shared.seats)
  shared.snapshot = globalSnapshot(gameSim)
  let router = buildRouter()
  gameServer = newServer(router, websocketHandler, workerThreads = 4)
  createThread(gameThread, runGame, runtimeConfig)
  echo "hidden-agenda: serving on ", runtimeConfig.host, ":",
    runtimeConfig.port
  gameServer.serve(Port(runtimeConfig.port), runtimeConfig.host)
