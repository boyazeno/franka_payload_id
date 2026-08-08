#!/usr/bin/env bash
# Run the test suite with a clean environment.
#
# A system ROS installation on PYTHONPATH puts its pytest plugins (launch_testing, ...)
# on the path of any venv, and they fail to import. Clearing PYTHONPATH and disabling
# plugin autoload avoids it. Inside the analysis container neither is necessary.
set -euo pipefail
cd "$(dirname "$0")/.."
exec env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest "$@"
