#!/usr/bin/env python3
from __future__ import annotations

import os
import sys


def main() -> int:
    if "--help" in sys.argv[1:]:
        print("usage: fake_setsid [--wait] command")
        return 0
    args = [value for value in sys.argv[1:] if value != "--wait"]
    if not args:
        return 2
    os.setsid()
    os.execvp(args[0], args)
    return 127


if __name__ == "__main__":
    sys.exit(main())
