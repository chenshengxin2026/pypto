# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Dynamic valid_shape examples — if/else and loop patterns.

Demonstrates DSL patterns where the valid length of a tile is computed
dynamically via if/else branches or loops, then used in a single
load+fillpad:

Pattern 1 (if/else)::

    if is_last:
        vlen = last_valid_len      # partial block
    else:
        vlen = full_len            # full block
    tile = pl.load(..., valid_shapes=[rows, vlen])
    padded = pl.tile.fillpad(tile, pad_value=PadValue.min)

Pattern 2 (loop + if/else)::

    for i in range(n_blocks):
        if i == n_blocks - 1:
            vlen = last_valid_len  # partial (last block)
        else:
            vlen = block_size      # full
        tile = pl.load(..., valid_shapes=[Q_TILE, vlen])
        padded = pl.tile.fillpad(tile, pad_value=PadValue.min)

Use ``build_if_else_program()`` and ``build_loop_program()`` to obtain
``@pl.program`` classes for these patterns.
"""

# pyright: reportUndefinedVariable=false

import pypto.language as pl

# Tile / tensor dimensions
Q_TILE = 64
BLOCK_COL = 64
N_ROW = 128  # sij_buf rows = Q_TILE * max_blocks(2)


# ── Shared InCore kernels ────────────────────────────────────────────────────


@pl.function(type=pl.FunctionType.InCore)
def kernel_dyn_valid_shape(
    data: pl.Tensor[[64, 64], pl.FP32],
    scale: pl.Scalar[pl.FP32],
    is_last: pl.Scalar[pl.BOOL],
    valid_len: pl.Scalar[pl.INDEX],
    full_len: pl.Scalar[pl.INDEX],
    output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
) -> pl.Tensor[[64, 64], pl.FP32]:
    """Load with dynamic valid_shape selected via if/else, fillpad, then scale."""
    if is_last:
        vlen: pl.Scalar[pl.INDEX] = valid_len
    else:
        vlen: pl.Scalar[pl.INDEX] = full_len
    s_tile: pl.Tile[[64, 64], pl.FP32] = pl.load(
        data, [0, 0], [64, 64], valid_shapes=[64, vlen], target_memory=pl.MemorySpace.Vec
    )
    s_padded: pl.Tile[[64, 64], pl.FP32] = pl.tile.fillpad(s_tile, pad_value=pl.PadValue.min)
    scaled: pl.Tile[[64, 64], pl.FP32] = pl.mul(s_padded, scale)
    out: pl.Tensor[[64, 64], pl.FP32] = pl.store(scaled, [0, 0], output)
    return out


@pl.function(type=pl.FunctionType.InCore)
def kernel_loop_dyn_valid(
    sij_buf: pl.Tensor[[N_ROW, BLOCK_COL], pl.FP32],
    scale: pl.Scalar[pl.FP32],
    n_blocks: pl.Scalar[pl.INDEX],
    last_valid_len: pl.Scalar[pl.INDEX],
    block_size: pl.Scalar[pl.INDEX],
    output: pl.Out[pl.Tensor[[N_ROW, BLOCK_COL], pl.FP32]],
) -> pl.Tensor[[N_ROW, BLOCK_COL], pl.FP32]:
    """Loop over blocks; last block uses partial valid_shape, others use full."""
    for i, (out,) in pl.range(n_blocks, init_values=(output,)):
        if i == n_blocks - 1:
            vlen: pl.Scalar[pl.INDEX] = last_valid_len
        else:
            vlen: pl.Scalar[pl.INDEX] = block_size
        s_tile: pl.Tile[[Q_TILE, BLOCK_COL], pl.FP32] = pl.load(
            sij_buf,
            [i * Q_TILE, 0],
            [Q_TILE, BLOCK_COL],
            valid_shapes=[Q_TILE, vlen],
            target_memory=pl.MemorySpace.Vec,
        )
        s_padded: pl.Tile[[Q_TILE, BLOCK_COL], pl.FP32] = pl.tile.fillpad(s_tile, pad_value=pl.PadValue.min)
        scaled: pl.Tile[[Q_TILE, BLOCK_COL], pl.FP32] = pl.mul(s_padded, scale)
        updated: pl.Tensor[[N_ROW, BLOCK_COL], pl.FP32] = pl.store(scaled, [i * Q_TILE, 0], out)
        loop_result = pl.yield_(updated)
    return loop_result


# ── Program builders ─────────────────────────────────────────────────────────


def build_if_else_program():
    """Build a program that selects valid_shape via if/else, then load+fillpad.

    Returns:
        A @pl.program class with an orchestration function that reads scalar
        configs from 1-element tensors and calls kernel_dyn_valid_shape.
    """

    @pl.program
    class DynValidShapeIfElse:
        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            data: pl.Tensor[[64, 64], pl.FP32],
            scale_cfg: pl.Tensor[[1], pl.FP32],
            flag_cfg: pl.Tensor[[1], pl.INT64],
            valid_len_cfg: pl.Tensor[[1], pl.INT64],
            full_len_cfg: pl.Tensor[[1], pl.INT64],
            output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            scale: pl.Scalar[pl.FP32] = pl.tensor.read(scale_cfg, [0])
            is_last: pl.Scalar[pl.INT64] = pl.tensor.read(flag_cfg, [0])
            valid_len: pl.Scalar[pl.INT64] = pl.tensor.read(valid_len_cfg, [0])
            full_len: pl.Scalar[pl.INT64] = pl.tensor.read(full_len_cfg, [0])
            output = kernel_dyn_valid_shape(data, scale, is_last, valid_len, full_len, output)
            return output

    return DynValidShapeIfElse


def build_loop_program():
    """Build a program that loops over blocks with dynamic valid_shape per iteration.

    Returns:
        A @pl.program class with an orchestration function that reads scalar
        configs from 1-element tensors and calls kernel_loop_dyn_valid.
    """

    @pl.program
    class LoopDynValid:
        @pl.function(type=pl.FunctionType.Orchestration)
        def orchestrator(
            self,
            sij_buf: pl.Tensor[[N_ROW, BLOCK_COL], pl.FP32],
            scale_cfg: pl.Tensor[[1], pl.FP32],
            n_blocks_cfg: pl.Tensor[[1], pl.INT64],
            last_valid_len_cfg: pl.Tensor[[1], pl.INT64],
            block_size_cfg: pl.Tensor[[1], pl.INT64],
            output: pl.Out[pl.Tensor[[N_ROW, BLOCK_COL], pl.FP32]],
        ) -> pl.Tensor[[N_ROW, BLOCK_COL], pl.FP32]:
            scale: pl.Scalar[pl.FP32] = pl.tensor.read(scale_cfg, [0])
            n_blocks: pl.Scalar[pl.INT64] = pl.tensor.read(n_blocks_cfg, [0])
            last_valid_len: pl.Scalar[pl.INT64] = pl.tensor.read(last_valid_len_cfg, [0])
            block_size: pl.Scalar[pl.INT64] = pl.tensor.read(block_size_cfg, [0])
            output = kernel_loop_dyn_valid(sij_buf, scale, n_blocks, last_valid_len, block_size, output)
            return output

    return LoopDynValid


if __name__ == "__main__":
    print("=== If/Else Dynamic Valid Shape ===")
    print(build_if_else_program().as_python())
    print("\n=== Loop Dynamic Valid Shape ===")
    print(build_loop_program().as_python())
