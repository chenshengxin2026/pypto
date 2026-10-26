# WSE Calendar FFN on A3

The `wse-sim-1026` branches implement an executable, synchronous subset of the
WSE Calendar design. [The example](../../../examples/models/wse_ffn.py) uses
ordinary PyPTO GEMM, vector activation and cast operations, plus
`pl.wse.allgather(x, group=pl.wse.Group((2, 5)), axis=1)`.

Four ordered logical workers `(2, 5, 9, 12)` own column shards. B groups are
`(2, 5)` and `(9, 12)`; C groups are `(2, 9)` and `(5, 12)`. With
`T=16, H=64, I=128`, each worker computes 32 columns of
`Z = FP16(SiLU(X @ W_gate) * (X @ W_up))`, gathers 64 then 128 columns, and
computes 16 output columns of `FP16(Z @ W_down)`. GEMMs accumulate in FP32.
The output remains four shards. There is no third collective.

## Reproduce from the four forks

Use Linux AArch64 with an A3, CANN 9.0 (`bisheng` on PATH), compatible
`torch`/`torch_npu`, Python 3.10+, a C++17 compiler, CMake and Ninja. PTOAS needs
LLVM19/MLIR from `vpto-dev/llvm-project`'s `feature-vpto` branch, built with
Python bindings and `pybind11<3`; follow its README to provision that toolchain.
The validated LLVM source revision is
`fa2fd1f75d742fec91ecf40d98c9c7961647ec42`. Set `LLVM_BUILD_DIR` to its build tree.
All device commands below go through `task-submit`.

```bash
mkdir wse-sim && cd wse-sim
for repo in pypto simpler PTOAS pto-isa; do
  git clone --branch wse-sim-1026 --single-branch \
    "https://github.com/chenshengxin2026/$repo.git" "$repo"
done
git -C pypto submodule update --init 3rdparty/libbacktrace 3rdparty/msgpack-c
python3 -m venv --system-site-packages pypto/.venv
source pypto/.venv/bin/activate
python -m pip install 'nanobind>=2,<3' 'pybind11<3' cmake ninja numpy pytest

cmake -S pypto -B pypto/build -DCMAKE_BUILD_TYPE=Release \
  -Dnanobind_DIR="$(python -m nanobind --cmake_dir)"
cmake --build pypto/build --parallel 2
cmake -S PTOAS -B PTOAS/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPTO_ENABLE_PYTHON_BINDING=ON -DBUILD_TESTING=OFF -DENABLE_PACKAGE=OFF \
  -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build PTOAS/build --parallel 2 --target PTOASPythonPackage
bash pypto/examples/models/run_wse_a3.sh "$(command -v python)"
```

PTOAS fetches `cann-cmake` at `master-055`. An offline installation may pass
`-DCANN_3RD_LIB_PATH=/path/to/deps`, containing `cann-cmake/`; the tested revision
is `a7ef9a7f04ea9efc5ad388cdb8cce0b0c1759c70` (check the supplied dependency's
revision when reproducing). The PyPTO runtime submodule is optional for this
adapter: it compiles against the sibling simpler checkout's actual Tensor ABI.
No simpler native extension is needed. The runtime submodule points to the
matching fork commit; its relative URL follows the PyPTO fork owner.
Source ZIPs work with the same layout;
populate PyPTO's two pinned third-party directories separately when using ZIPs.

The script produces `pypto/build_output/wse_ffn/`: per-pass IR, PTO IR,
PTOAS C++, unchanged PyPTO `kernel_entry` wrappers, compiled AIC/AIV libraries,
compiler signatures, build commands/logs, device logs and `validation.json`.
It reruns three seeds with both physical launch orders. A failure exits nonzero.

## Implemented lowering and execution contract

| Layer | Responsibility |
| --- | --- |
| PyPTO | Preserve ordered logical members through JIT/parser; lower Tensor to flat L1 Tile; plan non-aliasing storage with `MemoryPlanner.PYPTO`; finalize remote byte count after allocation; emit `pto.tcalendar` and a hidden context argument |
| PTOAS | Verify type, final valid geometry, expected bytes and static L1 ranges; preserve communication effects; emit `pto::wse::RunCalendar<axis, members...>` |
| PTO-ISA | `TCalendar<Plan>`: complete dependencies, BSP, arm receive, issue send, commit remote payload and local copy, then return synchronously |
| CCE A3 mock | Real cbuf/L1 endpoints and MTE DMA; launch-owned GM mailboxes emulate transport; epochs and per-source accounting detect invalid completion |
| simpler | Stub mapping and paired route/slot construction, install tables, reserve shared epochs, bind logical rows independently of physical order, launch resident groups and check status |

`exp_val = (group.size - 1) * rows * cols * 2` counts remote valid bytes only.
The receiver's address capacity includes the local slice. The receive intrinsic
arms without waiting; each core can issue SEND before waiting for payload.
The synchronous facade completes all DMA before releasing its L1 storage.
The compiler retains a collective even when its result is unused.

Each group has a separate deployment and BSP handle. Tables install once per
deployment; every invocation owns an epoch, mailboxes, signals and status until
stream completion. The paired stub tables represent a complete multicast
audience and deterministic slots. They do not model NoC timing or arbitration.

A3 has no native Calendar intrinsics or persistent L1 across these launches.
The adapter therefore uses ordinary GM-backed DMA transport and explicitly
spills between A/B/C/D kernels. B/C groups execute sequentially; each individual
group's participants are resident together. This validates compilation,
communication semantics and FFN arithmetic on real A3, without claiming native
WSE throughput, routing feasibility or a 32-worker residency model.

Supported: static rank-2 FP16 AllGather, axes 0/1 (including negative frontend
axes), packed 32-byte source rows, 1–24 members, receive capacity ≤128 KiB,
one unconditional collective per standalone kernel, A3 and PYPTO memory
planning. Asynchronous collectives, reduction, dynamic/tail geometry, multiple
Calendar calls per kernel, PTOAS memory planning and ordinary ChipWorker
orchestration bindings are unsupported and must use a future extension.
Compile standalone `@pl.jit.incore` kernels and execute them with `simpler.wse`.

## Validation

Communication is compared bit-for-bit against the concatenation of the actual
device-produced shards, independently of the FFN numeric oracle. CPU FP32
matmul/SiLU with FP16 rounding supplies the full FFN reference. The test checks
every intermediate gather, logical ordering, epoch, status and byte count;
final accuracy uses `rtol=0.002, atol=0.0001`. Axis 0, unordered sparse IDs and
the singleton's zero-byte receive contract are also executed on the board.

```bash
PYTHONPATH=pypto/python python -m pytest pypto/tests/ut/language/test_wse.py -q
python PTOAS/test/wse_calendar.py --ptoas PTOAS/build/tools/ptoas/ptoas
g++ -std=c++17 -Wall -Wextra -Werror -Ipto-isa/include \
  pto-isa/tests/cpu/wse/calendar_contract.cpp -o /tmp/wse-contract
/tmp/wse-contract
PYTHONPATH=simpler/python python -m pytest \
  --confcutdir=simpler/tests/ut/py/wse simpler/tests/ut/py/wse -q
```

The validation structure follows layered contract, transport and numerical
checks, as in [pto-isa PR #252](https://github.com/hw-native-sys/pto-isa/pull/252).
The instruction ABI and FFN data distribution follow the Calendar design.

The [recorded A3 validation](../../wse-a3-validation.json) passed all six runs
(seeds 1026–1028, both mappings) on Ascend910_9392. Maximum absolute FFN
error was `3.0517578125e-5`; every gather was bit exact. B/C remote committed
bytes were 1024/2048 per member; axis 0 committed 64 and singleton 0.
The report records task IDs, device/tool versions, epochs and artifact hashes.
