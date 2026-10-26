#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -euo pipefail

# The four branch checkouts are siblings: pypto, simpler, PTOAS, pto-isa.
pypto_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
repos_root=$(dirname -- "$pypto_root")
python_bin=$(command -v "${1:-python}")
artifact="$pypto_root/build_output/wse_ffn"
runner="$repos_root/simpler/tests/st/a2a3/wse/run_ffn.py"
export PYTHONPATH="$pypto_root/python:$repos_root/simpler/python${PYTHONPATH:+:$PYTHONPATH}"
export PTOAS_ROOT="$repos_root/PTOAS/build/tools/ptoas"

"$python_bin" "$pypto_root/examples/models/wse_ffn.py" --output "$artifact"
"$python_bin" "$runner" build --artifact "$artifact" --isa-root "$repos_root/pto-isa"

printf -v precheck '%q ' bash "$repos_root/simpler/.claude/skills/onboard-arch-precheck/check.sh" a2a3
task-submit --run "$precheck"
mkdir -p "$artifact/ascend"
export ASCEND_PROCESS_LOG_PATH="$artifact/ascend"
printf -v command '%q ' "$python_bin" "$runner" run --artifact "$artifact" --repeats 3
task-submit --device auto --max-time 300 --run "$command --device \"\$TASK_DEVICE\""
