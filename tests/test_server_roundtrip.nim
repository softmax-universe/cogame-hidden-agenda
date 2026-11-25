## End-to-end public host + player round-trip tests covering:
## 1. Regular bare JSON and Markdown-fenced JSON from a native model seat
##    actually executing through the real server and producing scored results.
## 2. Invalid / ambiguous multi-object model output recording a player_error
##    (COGAME_PLAYER_FAILURE_URI) without retrying and terminating unscored
##    without results.json.
## 3. Provider HTTP failure (503) recording episode_inconclusive
##    (COGAME_PLAYER_FAILURE_URI) for the interrupted seat without retrying and
##    terminating unscored without results.json.
## 4. Missing LLM credentials recording episode_inconclusive
##    (COGAME_PLAYER_FAILURE_URI) for the interrupted seat and terminating
##    unscored without results.json.

import std/[os, osproc, strutils]

const
  RepoRoot = currentSourcePath().parentDir().parentDir()
  RoundTripScript = """
import http.server
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

game_bin, player_bin = sys.argv[1], sys.argv[2]


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_mock_llm(mode):
    calls = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            slot = self.headers.get("X-Coworld-Player-Slot")
            calls.append({"path": self.path, "slot": slot, "body": json.loads(raw)})
            idx = len(calls)
            if mode == "valid":
                if idx == 1:
                    text = json.dumps({
                        "plan": [{"job": "mine", "at": "S2"}, {"job": "deposit"}],
                        "hunch": "bare json opening",
                        "notes": "call 1",
                    })
                else:
                    text = "```json\n" + json.dumps({
                        "plan": [{"job": "guard"}],
                        "vote": "skip",
                        "hunch": "fenced json meeting",
                        "notes": "call 2",
                    }) + "\n```"
                payload = json.dumps({
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": text}],
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
            elif mode == "ambiguous":
                text = '{"plan":[{"job":"guard"}]} {"plan":[{"job":"hold"}]}'
                payload = json.dumps({
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": text}],
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
            elif mode == "provider_503":
                payload = json.dumps({"error": "upstream unavailable"}).encode("utf-8")
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_response(500)
                self.end_headers()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}", calls


def run_scenario(name, llm_mode):
    work = Path(tempfile.mkdtemp(prefix=f"ha-rt-{name}-"))
    port = free_port()
    config = {
        "tokens": [f"t{i}" for i in range(5)],
        "players": [{"name": f"P{i}"} for i in range(5)],
        "seed": 5,
        "impostorSlot": 4,
        "variant": "hidden-agenda-notalk",
        "chat": False,
        "meetingTicks": 25,
        "sayTick": -1,
        "revealTick": 5,
        "switchTick": 18,
        "resolveTick": 23,
        "maxTicks": 260,
        "meetingCadenceTicks": 60,
        "depositTarget": 4,
        "minBatchSeconds": 0,
        "llmTimeoutSeconds": 5,
        "playerConnectTimeoutSeconds": 10,
        "shutdownGraceSeconds": 0,
    }
    config_path = work / "config.json"
    results_path = work / "results.json"
    replay_path = work / "replay.json"
    failure_path = work / "player_failure.json"
    log_path = work / "game_error.log"
    config_path.write_text(json.dumps(config))

    mock_server = None
    calls = []
    game_env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "COGAME_HOST": "127.0.0.1",
        "COGAME_PORT": str(port),
        "COGAME_CONFIG_URI": f"file://{config_path}",
        "COGAME_RESULTS_URI": f"file://{results_path}",
        "COGAME_SAVE_REPLAY_URI": f"file://{replay_path}",
        "COGAME_PLAYER_FAILURE_URI": f"file://{failure_path}",
        "COGAME_LOG_URI": f"file://{log_path}",
    }
    if llm_mode != "missing_creds":
        mock_server, endpoint, calls = start_mock_llm(llm_mode)
        game_env["COWORLD_LLM_ENDPOINT"] = endpoint
        game_env["COWORLD_LLM_MODEL"] = "anthropic/claude-haiku-4.5"

    game_proc = subprocess.Popen(
        [game_bin],
        env=game_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    player_procs = []
    try:
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=0.5) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                time.sleep(0.05)
        else:
            raise AssertionError(f"{name}: server never became healthy")

        for slot in range(5):
            penv = {
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", ""),
                "COWORLD_PLAYER_WS_URL": f"ws://127.0.0.1:{port}/player?slot={slot}&token=t{slot}",
            }
            if slot == 0:
                penv["COWORLD_LLM_ENABLED"] = "true"
            elif slot == 4:
                penv["PLAYER_SCRIPTED"] = "lurker"
            else:
                penv["PLAYER_SCRIPTED"] = "miner"
            player_procs.append(subprocess.Popen(
                [player_bin],
                env=penv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            ))

        game_out, _ = game_proc.communicate(timeout=30)
        rc = game_proc.returncode
        for p in player_procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        return {
            "rc": rc,
            "stdout": game_out,
            "calls": calls,
            "results": json.loads(results_path.read_text()) if results_path.exists() else None,
            "replay": json.loads(replay_path.read_text()) if replay_path.exists() else None,
            "failure": json.loads(failure_path.read_text()) if failure_path.exists() else None,
            "log": log_path.read_text() if log_path.exists() else None,
        }
    finally:
        if game_proc.poll() is None:
            game_proc.kill()
        for p in player_procs:
            if p.poll() is None:
                p.kill()
        if mock_server is not None:
            mock_server.shutdown()


# 1. Valid bare JSON + fenced JSON round-trip through real server + native player
valid = run_scenario("valid", "valid")
assert valid["rc"] == 0, f"valid scenario failed rc={valid['rc']}\n{valid['stdout']}"
assert valid["failure"] is None, f"unexpected player_failure: {valid['failure']}"
assert valid["results"] is not None, "valid scenario must write results.json"
assert valid["replay"] is not None, "valid scenario must write replay.json"
assert len(valid["calls"]) >= 2, f"expected >=2 LLM calls (bare + fenced), got {len(valid['calls'])}"
assert all(c["slot"] == "0" for c in valid["calls"]), f"wrong slot attribution: {valid['calls']}"
seat0_orders = [e for e in valid["replay"]["events"] if e.get("k") == "order" and e.get("seat") == 0]
assert len(seat0_orders) >= 2, f"expected >=2 seat-0 orders, got {seat0_orders}"
assert all(o.get("source") == "llm" for o in seat0_orders), f"expected llm source: {seat0_orders}"
assert seat0_orders[0]["plan"][0]["job"] == "mine" and seat0_orders[0]["plan"][0]["at"] == "S2"
assert seat0_orders[1]["plan"][0]["job"] == "guard"
print("roundtrip 1 (bare + fenced JSON): ok")

# 2. Invalid/ambiguous multi-object response -> player_error, no retry, unscored (no results.json)
ambig = run_scenario("ambiguous", "ambiguous")
assert ambig["rc"] != 0, f"ambiguous scenario must exit non-zero\n{ambig['stdout']}"
assert ambig["results"] is None, f"ambiguous failure must not write numeric results: {ambig['results']}"
assert ambig["replay"] is None, "ambiguous failure must not write authoritative replay"
assert ambig["failure"] is not None, "ambiguous failure must write player_failure.json"
assert set(ambig["failure"].keys()) == {"failed_policy_index", "message", "kind"}, f"unexpected keys: {ambig['failure']}"
assert ambig["failure"]["failed_policy_index"] == 0, f"expected slot 0 failure: {ambig['failure']}"
assert ambig["failure"]["kind"] == "player_error", f"expected player_error kind: {ambig['failure']}"
assert len(ambig["calls"]) == 1, f"expected 1 call without retry, got {len(ambig['calls'])}"
print("roundtrip 2 (ambiguous multi-object): ok")

# 3. Provider HTTP 503 failure -> episode_inconclusive, no retry, unscored
prov = run_scenario("provider_503", "provider_503")
assert prov["rc"] != 0, f"provider 503 scenario must exit non-zero\n{prov['stdout']}"
assert prov["results"] is None, f"provider failure must not write numeric results: {prov['results']}"
assert prov["replay"] is None, "provider failure must not write authoritative replay"
assert prov["failure"] is not None, "provider 503 must write player_failure.json with episode_inconclusive"
assert set(prov["failure"].keys()) == {"failed_policy_index", "message", "kind"}, f"unexpected keys: {prov['failure']}"
assert prov["failure"]["failed_policy_index"] == 0, f"expected interrupted slot 0: {prov['failure']}"
assert prov["failure"]["kind"] == "episode_inconclusive", f"expected episode_inconclusive kind: {prov['failure']}"
assert len(prov["calls"]) == 1, f"expected 1 call without retry, got {len(prov['calls'])}"
assert prov["log"] is not None and "infrastructure failure" in prov["log"], f"missing infra log: {prov['log']}"
print("roundtrip 3 (provider 503 episode_inconclusive): ok")

# 4. Missing credentials -> episode_inconclusive, unscored
creds = run_scenario("missing_creds", "missing_creds")
assert creds["rc"] != 0, f"missing creds scenario must exit non-zero\n{creds['stdout']}"
assert creds["results"] is None, f"missing creds must not write numeric results: {creds['results']}"
assert creds["replay"] is None, "missing creds must not write authoritative replay"
assert creds["failure"] is not None, "missing creds must write player_failure.json with episode_inconclusive"
assert set(creds["failure"].keys()) == {"failed_policy_index", "message", "kind"}, f"unexpected keys: {creds['failure']}"
assert creds["failure"]["failed_policy_index"] == 0, f"expected interrupted slot 0: {creds['failure']}"
assert creds["failure"]["kind"] == "episode_inconclusive", f"expected episode_inconclusive kind: {creds['failure']}"
assert creds["log"] is not None and "infrastructure failure" in creds["log"], f"missing infra log: {creds['log']}"
print("roundtrip 4 (missing credentials episode_inconclusive): ok")
"""

proc mustRun(cmd: string) =
  let (output, code) = execCmdEx(cmd)
  if code != 0:
    echo "FAIL: command exited ", code, ": ", cmd
    echo output
    quit(1)
  if output.strip().len > 0:
    echo output.strip()

proc main() =
  let tmpDir = getTempDir() / ("ha-roundtrip-" & $getCurrentProcessId())
  createDir(tmpDir)
  defer:
    try:
      removeDir(tmpDir)
    except CatchableError:
      discard

  let gameBin = tmpDir / "hidden-agenda"
  let playerBin = tmpDir / "hidden-agenda-player"
  let scriptPath = tmpDir / "roundtrip.py"
  writeFile(scriptPath, RoundTripScript)

  mustRun("nim c --hints:off --path:" & quoteShell(RepoRoot / "src") &
    " -o:" & quoteShell(gameBin) & " " &
    quoteShell(RepoRoot / "src" / "hidden_agenda.nim"))
  mustRun("nim c --hints:off --path:" & quoteShell(RepoRoot / "src") &
    " -o:" & quoteShell(playerBin) & " " &
    quoteShell(RepoRoot / "src" / "hidden_agenda_player.nim"))
  mustRun("python3 " & quoteShell(scriptPath) & " " &
    quoteShell(gameBin) & " " & quoteShell(playerBin))

  echo "test_server_roundtrip: ok"

main()
