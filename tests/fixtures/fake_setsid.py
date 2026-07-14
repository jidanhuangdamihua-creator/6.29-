#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys


def main() -> int:
    if "--help" in sys.argv[1:]:
        print("usage: fake_setsid [--wait] command")
        return 0
    args = [value for value in sys.argv[1:] if value != "--wait"]
    if not args:
        return 2
    completed = subprocess.run(args, check=False, start_new_session=True)
    return int(completed.returncode)


if __name__ == "__main__":
    sys.exit(main())
