#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Run a command with a hard timeout. Exit 124 on timeout.")
    parser.add_argument("--timeout", type=int, required=True, help="Timeout in seconds")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run")
    args = parser.parse_args()
    if not args.cmd:
        print("ERROR: no command provided", file=sys.stderr); return 2
    cmd = args.cmd
    if cmd and cmd[0] == "--": cmd = cmd[1:]
    use_setsid = hasattr(os, "setsid")
    try:
        proc = subprocess.Popen(cmd, preexec_fn=os.setsid if use_setsid else None)
        process_group_id = os.getpgid(proc.pid) if use_setsid else None
        return proc.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print(f"\n[codex_timeout] TIMEOUT after {args.timeout}s: {' '.join(cmd)}", file=sys.stderr)
        try:
            if process_group_id is not None: os.killpg(process_group_id, signal.SIGTERM)
            else: proc.terminate()
        except ProcessLookupError: pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired: pass
        if process_group_id is not None:
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError: pass
        elif proc.poll() is None:
            proc.kill()
        if proc.poll() is None:
            proc.wait()
        return 124

if __name__ == "__main__": raise SystemExit(main())
