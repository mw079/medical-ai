# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "sdbench[cpt-costing] @ git+https://github.com/microsoft/sdbench@304ce9625a046cea97d64e3b1ba5d19217b5095c",
#   "cursor-sdk==1.4.2", "pyarrow==24.0.0", "python-dotenv==1.1.0",
# ]
# ///
"""Run curation and benchmark regression tests with network calls blocked."""

from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / name) for name in ("curation", "evaluation", "experiments", "tests")]
sys.path.insert(0, str(ROOT))


def main():
    def network_blocked(*_args, **_kwargs):
        raise AssertionError("Offline tests must not connect to the network")

    # SDBench retries failed model calls with real backoff delays. Tests that expect a failure
    # would otherwise wait for them.
    with (patch("time.sleep"),
          patch.object(socket.socket, "connect", network_blocked),
          patch.object(socket.socket, "connect_ex", network_blocked),
          patch.object(socket, "create_connection", network_blocked)):
        if sys.argv[1:2] == ["--audit-local"]:
            from audit_local_data import main as audit
            return audit(sys.argv[2:])
        if sys.argv[1:]:
            raise SystemExit("Use --audit-local to audit existing data")
        suite = unittest.TestLoader().discover(str(ROOT / "tests"), pattern="test_*.py")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
