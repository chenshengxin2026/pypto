# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""WSE frontend and the complete standard pass/codegen pipeline, without a device."""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.language.parser.diagnostics import InvalidOperationError


def gather_kernel(group, axis=1):
    rows, cols = (2 * group.size, 16) if axis == 0 else (2, 16 * group.size)

    @pl.jit.incore
    def gather(x: pl.Tensor[[2, 16], pl.FP16], y: pl.Out[pl.Tensor[[rows, cols], pl.FP16]]):
        value = pl.wse.allgather(x, group=group, axis=axis)
        y = pl.assemble(y, value, [0, 0])
        return y

    return gather


@pytest.mark.parametrize("members", [(), (1, 1), (-1,), (True,), (2**31,), [1, 2]])
def test_bad_group(members):
    with pytest.raises(ValueError):
        pl.wse.Group(members)


@pytest.mark.parametrize("members,axis", [((9, 2), 1), ((9, 2), 0), ((17,), 1)])
def test_compile_to_calendar(tmp_path, members, axis):
    ir.compile(
        gather_kernel(pl.wse.Group(members), axis).specialize(),
        output_dir=str(tmp_path),
        skip_ptoas=True,
    )
    text = "\n".join(path.read_text() for path in tmp_path.rglob("*.pto"))
    assert '"pto.tcalendar"' in text
    assert "%__wse_context: i64" in text
    assert "members = array<i64: " + ", ".join(map(str, members)) + ">" in text
    assert f"arith.constant {(len(members) - 1) * 64} : i32" in text
    assert "slayout = none_box" in text or "slayout=none_box" in text


def test_reject_unplanned_storage(tmp_path):
    with (
        passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS),
        pytest.raises(ValueError, match=r"requires MemoryPlanner.PYPTO"),
    ):
        ir.compile(
            gather_kernel(pl.wse.Group((2, 9))).specialize(),
            output_dir=str(tmp_path),
            skip_ptoas=True,
        )


def test_unused_calendar_is_observable(tmp_path):
    group = pl.wse.Group((2, 9))

    @pl.jit.incore
    def two_calls(x: pl.Tensor[[2, 16], pl.FP16], y: pl.Out[pl.Tensor[[2, 32], pl.FP16]]):
        unused = pl.wse.allgather(x, group=group, axis=1)  # noqa: F841 - communication remains observable
        value = pl.wse.allgather(x, group=group, axis=1)
        y = pl.assemble(y, value, [0, 0])
        return y

    with pytest.raises(ValueError, match="one unconditional Calendar call"):
        ir.compile(two_calls.specialize(), output_dir=str(tmp_path), skip_ptoas=True)


def test_reject_unaligned_input():
    group = pl.wse.Group((2, 9))

    @pl.jit.incore
    def bad(x: pl.Tensor[[2, 7], pl.FP16]):
        return pl.wse.allgather(x, group=group, axis=1)

    with pytest.raises(InvalidOperationError, match="aligned to 32"):
        bad.specialize()


def test_reject_oversized_input():
    group = pl.wse.Group((2, 9))

    @pl.jit.incore
    def bad(x: pl.Tensor[[2**40, 16], pl.FP16]):
        return pl.wse.allgather(x, group=group, axis=1)

    with pytest.raises(InvalidOperationError, match="dimensions in 1..65536"):
        bad.specialize()


def test_reject_unbound_chip_orchestration(tmp_path):
    group = pl.wse.Group((2, 9))

    @pl.program
    class UnboundCalendar:
        @pl.function(type=pl.FunctionType.InCore)
        def gather(self, x: pl.Tensor[[2, 16], pl.FP16], y: pl.Out[pl.Tensor[[2, 32], pl.FP16]]):
            value = pl.wse.allgather(x, group=group, axis=1)
            y = pl.assemble(y, value, [0, 0])
            return y

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[2, 16], pl.FP16]) -> pl.Tensor[[2, 32], pl.FP16]:
            y = pl.create_tensor([2, 32], dtype=pl.FP16)
            return self.gather(x, y)

    with pytest.raises(ValueError, match="requires simpler.wse deployment bindings"):
        ir.compile(UnboundCalendar, output_dir=str(tmp_path), skip_ptoas=True)
