# Experimental WSE Calendar frontend for A3

`pypto.wse` is a static graph frontend for the WSE Calendar A3 simulation.
It implements the logical programming contract in the 1014 WSE design:
column sharding, replica provenance, group size, two AllGathers, and a
column-sharded result. It emits native PTO IR for the modified PTOAS compiler.

This is an experimental frontend extension, separate from the existing
`pypto.language` parser and `pl.jit` pass pipeline. The design's proposed
`logical_parallel`, `dist.local_shard`, and `dist.sharded_result` syntax is
expressed using `Program(workers=...)`, `Program.input(layout=...)`, and
`Program.result(...)`. Existing `pl.jit` programs do not acquire WSE support
implicitly. See `examples/models/wse_calendar_ffn.py` for the complete FFN.

## Compilation contract

- `Replicated()` creates one private copy per logical worker.
- `ColumnSharded(parts=32)` places contiguous column intervals on workers.
- `Group(size=8)` describes a group size, not a physical mesh direction.
- `all_gather(..., axis=-1)` infers ordered members from intervals and replica
  provenance. The second gather uses replicas of different intervals rather
  than gathering identical copies of the first interval.
- `result(...)` exports logical column shards without inserting a third gather.

For the original FFN, `T=8`, `H=7168`, `I=3072`, and `workers=32`.
The local intermediates are `[8,96]`, `[8,768]`, `[8,3072]`, and `[8,224]`.
GEMMs accumulate FP16 inputs in FP32. SiLU has no clamp; its product with the
up projection is rounded to FP16 before the down projection.

`Program.emit(directory, nodes=48)` writes:

| File | Meaning |
| --- | --- |
| `program.json` | Static graph, layouts, and local tensor shapes |
| `manifest.json` | Logical groups, receive geometry, dependencies, memory budgets |
| `binding_schema.json` | Hidden deployment operands and device status layout |
| `op_*.pto` | Native `pto.wse_local` / `pto.tcalendar` operations |

`Program.compile(directory, ptoas=..., nodes=48)` runs the supplied native
PTOAS CLI for every kernel. It records the command, exit status, and diagnostic
log, and fails if any operation is rejected. Generated C++ is compiled by
`simpler.wse.build_device_library` against the modified PTO-ISA headers.

PG, physical mapping, routes, and slot images are absent from these compilation
inputs. `nodes` selects a target ABI, not a particular machine's PG snapshot.
Deployment can reuse one binary under different mappings and defect maps.

## Supported subset and resource constraints

- Static two-dimensional FP16 inputs, FP32 GEMM/SiLU/product, FP32-to-FP16 cast.
- GEMM: `1 <= M <= 16`, `K` a multiple of 64 up to 65536, local `N` a multiple
  of 16 up to 4096. The current deployment stub supports 32 workers.
- Equal-width AllGather on the feature axis, with 32-byte aligned source rows.
  Unsupported or ambiguous layouts are rejected.
- Private local DRAM allocations use 2 KiB pages and must fit 384 MiB per
  worker. A3 backing allocations are also aligned to 2 KiB.
- Calendar send/receive UB regions must be disjoint. Scratch high-water marks
  obey both the WSE budgets and the smaller A3 capacities.
- Default route ABI 3 uses 96 route bits for the whitepaper's 6x8 topology.
  `nodes=40` explicitly selects the older 80-bit ABI 2. Both entries are 16
  bytes, with the receive count at offset 12; ABI 3 omits the redundant ABI 2
  landing-count and flags bytes. These are simulation ABI choices, not a
  resolution of the real hardware specification's open topology question.

The graph builder does not implement dynamic shapes, general partitioning,
Reduce/AllReduce, native WSE instructions, or NoC timing simulation. The A3
transport adapter uses private GM backing slots for local DRAM and models
peer delivery with native PTO DMA/vector operations. CPU golden computation
is kept outside the execution path.

## Tests

The frontend tests in `tests/ut/wse/test_frontend.py` cover original FFN
geometry, replica ordering, instance-independent emission, invalid groups,
unsupported GEMM shapes, and UB exhaustion. Cross-repository validation also
checks native PTOAS diagnostics, deployment rejection paths, actual A3 output,
device receive-byte counters, and both gather snapshots against their input
shards bit for bit.
