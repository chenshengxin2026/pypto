# FinalizeWseCalendar

Finalizes the synchronous WSE Calendar contract after storage planning and
`MaterializeValidShapeSymbols`, before `LowerTileToBuffer`.

The pass accepts one unconditional `tile.wse_allgather` per standalone kernel.
It rejects collectives in loops or conditional branches and requires
`MemoryPlanner.PYPTO`. Both operands must have allocated, row-major, unboxed
L1 storage. The operation forbids send/receive aliasing in the memory planner;
PTOAS subsequently verifies concrete nonoverlapping address ranges and final
valid geometry.

The pass attaches `exp_bytes = (members - 1) * rows * cols * sizeof(FP16)`.
Code generation requires this finalized attribute and checks it against the
tile geometry before emitting `pto.tcalendar`. The local slice occupies
receive capacity but contributes no remote bytes. Programs without Calendar
operations are unchanged. The traversal is linear in the IR size.

See [WSE FFN on A3](../wse-a3.md) for the frontend, deployment and validation
contract. The C++ entry is `ir::pass::FinalizeWseCalendar()` and the Python entry
is `passes.finalize_wse_calendar()`.
