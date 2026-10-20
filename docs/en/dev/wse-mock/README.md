# WSE mock: PyPTO frontend

This branch adds the experimental `pypto.wse` frontend for the WSE Calendar
FFN simulation on real A3 hardware. Its upstream baseline is
`d06a0302f5b204acbc4102820b0a7e3edca1814d`. The PR targets the fork's `wse-mock`
branch; review code is on `wse-mock-calendar-a3`.

## Changes and interfaces

| File | Role |
| ---- | ---- |
| [Frontend](../../../../python/pypto/wse/__init__.py) | Static graph, column ownership, replica provenance, resource checks, native PTO emission |
| [FFN example](../../../../examples/models/wse_calendar_ffn.py) | Gate/up, SiLU/product, two AllGathers, down projection, sharded output |
| [Frontend tests](../../../../tests/ut/wse/test_frontend.py) | Positive and negative shape, group, memory, and emission contracts |

The design's `logical_parallel/local_shard/sharded_result` concepts are
expressed as `Program(workers)/Program.input(layout)/Program.result`.
`Group(size=8)` and `Group(size=4)` specify logical groups; replica provenance
determines their members. This extension is separate from the existing
`pl.jit` parser and pass pipeline.

`Program.compile(output, ptoas=..., nodes=48)` writes the graph, manifest,
binding schema, PTO IR, native PTOAS diagnostics, and generated C++. Physical
PG, mapping, routes, and slots are runtime bindings. The same binary can be
redeployed without recompilation. The [interface reference](../wse-calendar-a3.md)
describes supported shapes and the explicit 40/48-node ABI profiles.

## Build and run

Use the matching `wse-mock-calendar-a3` branches of
[PTOAS](https://github.com/chenshengxin2026/PTOAS/tree/wse-mock-calendar-a3),
[pto-isa](https://github.com/chenshengxin2026/pto-isa/tree/wse-mock-calendar-a3),
and [simpler](https://github.com/chenshengxin2026/simpler/tree/wse-mock-calendar-a3).
After the PRs merge, use all four `wse-mock` branches together. Build this
checkout following the root README; initialize its submodules first.

From this checkout, with the local native PyPTO extension built:

```bash
export PYTHONPATH="$PWD/python:$PWD/examples/models${PYTHONPATH:+:$PYTHONPATH}"
source .claude/skills/testing/load-env.sh
python -m pytest -q --noconftest -c /dev/null -n "$PYPTO_TEST_JOBS" \
  tests/ut/wse/test_frontend.py
```

The [cross-repository run guide](https://github.com/chenshengxin2026/simpler/blob/wse-mock-calendar-a3/docs/wse-mock/README.md)
includes portable FFN build and A3 golden-validation commands. Use the modified
native PTOAS executable and PTO-ISA checkout explicitly.

## Validation and limits

The four-repository implementation passed 46 targeted CPU/PTOAS tests plus
the C++ receiver contract, and 21 FFN executions on Ascend910_9392 on
2026-10-15. The original case is `T=8,H=7168,I=3072,P=32`; two seeds, three
deployments, and two repeats passed. Maximum absolute output error was
`0.015625`, maximum relative RMS error `8.2343e-5`, with pointwise
`atol=0.002, rtol=0.01` and relative RMS `<0.002`.

Both gathers match device input shards bit for bit at every worker. Outputs
are bit-identical across mappings/repeats for a fixed seed. These exactness
checks are distinct from the tolerance comparison with CPU golden.

The supported subset is static 2D FFN with 32 logical workers. PG/Batcher/CCE
mechanisms are mocked; mapping and NoC planning are stubs. Local DRAM uses
2 KiB pages with a 384 MiB per-worker limit; scratch respects both WSE and
A3 capacities. This validates functional execution and modeled constraints,
without claiming real WSE NoC timing or performance.
