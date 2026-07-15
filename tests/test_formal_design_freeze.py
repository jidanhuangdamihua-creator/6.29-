from hashlib import sha256
from pathlib import Path


def test_authoritative_design_is_frozen():
    path = Path("docs/superpowers/specs/2026-07-15-d1-d6-experiment-sealing-design.md")
    assert sha256(path.read_bytes()).hexdigest() == (
        "914ab6e4b3ac2eca7d2bb1c7cc2811a75c905995269b15b3300b0038f7343f6d"
    )
