#!/usr/bin/env bash
set -euo pipefail

export QHOME="/Users/oscarbonnet/PycharmProjects/q"
export QLIC="/Users/oscarbonnet/PycharmProjects/q"

exec "$QHOME/m64/q" scripts/smoke.q
