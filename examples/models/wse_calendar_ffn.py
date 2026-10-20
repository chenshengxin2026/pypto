# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""WSE FFN graph: local compute, two logical AllGathers, sharded output."""

from pypto import wse as pl


def build_ffn(tokens: int = 8, hidden: int = 7168, intermediate: int = 3072) -> pl.Program:
    program = pl.Program(workers=32)
    x = program.input("X", (tokens, hidden), layout=pl.Replicated())
    wg = program.input("W_gate", (hidden, intermediate), layout=pl.ColumnSharded(32))
    wu = program.input("W_up", (hidden, intermediate), layout=pl.ColumnSharded(32))
    wd = program.input("W_down", (intermediate, hidden), layout=pl.ColumnSharded(32))
    gate = pl.matmul(x, wg, acc_dtype=pl.FP32)
    up = pl.matmul(x, wu, acc_dtype=pl.FP32)
    z = pl.cast(pl.silu(gate) * up, pl.FP16)
    z_block = pl.all_gather(z, group=pl.Group(size=8), axis=-1)
    z_full = pl.all_gather(z_block, group=pl.Group(size=4), axis=-1)
    y = pl.cast(pl.matmul(z_full, wd, acc_dtype=pl.FP32), pl.FP16)
    program.result(y, axis=-1)
    return program
