# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration test for dynamic valid_shape in a loop with if/else branches.

Verifies the PTO-level pattern from the paged-attention design discussion:

  tile = alloc_tile<row=R, col=C, v_row=?, v_col=?, pad=min>
  for i in range(n_blocks):
      if i == n_blocks - 1:
          set_validshape(tile, vrow1, vcol1)   # partial (last block)
      else:
          set_validshape(tile, vrow2, vcol2)   # full

At the DSL level this translates to computing vlen in the if/else, then
performing a single load+fillpad(pad_value=min) with that computed length.

Test scenarios:
  1. n_blocks=2: block 0 is full (64 cols), block 1 is partial (48 valid cols)
  2. n_blocks=1: single block that is also the last → partial (48 valid cols)
"""

from typing import Any

import pytest
import torch
from examples.kernels.dyn_valid_shape import BLOCK_COL, N_ROW
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

# ---------------------------------------------------------------------------
# Test case 1: 2 blocks — block 0 full, block 1 partial (48 valid cols)
# ---------------------------------------------------------------------------


class LoopDynValidTwoBlocksTestCase(PTOTestCase):
    """n_blocks=2, block_size=64, last_valid_len=48.

    Expected:
      rows 0-63  (block 0, full):    input * scale
      rows 64-127 (block 1, last):   cols 0-47 = input * scale, cols 48-63 = -inf
    """

    __test__ = False

    def get_name(self) -> str:
        return "loop_dyn_valid_two_blocks"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("sij_buf", [N_ROW, BLOCK_COL], DataType.FP32, init_value=torch.randn),
            TensorSpec("scale_cfg", [1], DataType.FP32, init_value=2.0),
            TensorSpec(
                "n_blocks_cfg",
                [1],
                DataType.INT64,
                init_value=torch.tensor([2], dtype=torch.int64),
            ),
            TensorSpec(
                "last_valid_len_cfg",
                [1],
                DataType.INT64,
                init_value=torch.tensor([48], dtype=torch.int64),
            ),
            TensorSpec(
                "block_size_cfg",
                [1],
                DataType.INT64,
                init_value=torch.tensor([64], dtype=torch.int64),
            ),
            TensorSpec("output", [N_ROW, BLOCK_COL], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        from examples.kernels.dyn_valid_shape import build_loop_program  # noqa: PLC0415

        return build_loop_program()

    def compute_expected(self, tensors: dict[str, torch.Tensor], _params=None) -> None:
        scale = float(tensors["scale_cfg"][0].item())
        data = tensors["sij_buf"].clone()
        expected = torch.full((128, 64), float("-inf"), dtype=torch.float32)
        # Block 0 (full): all 64 cols valid
        expected[:64, :] = data[:64, :] * scale
        # Block 1 (last): cols 0-47 valid, cols 48-63 = -inf (pad.min * scale = -inf)
        expected[64:, :48] = data[64:, :48] * scale
        tensors["output"][:] = expected


# ---------------------------------------------------------------------------
# Test case 2: 1 block — single block is also the last → partial valid
# ---------------------------------------------------------------------------


class LoopDynValidOneBlockTestCase(PTOTestCase):
    """n_blocks=1, block_size=64, last_valid_len=48.

    Expected:
      rows 0-63 (block 0, also last): cols 0-47 = input * scale, cols 48-63 = -inf
      rows 64-127: untouched (zero-initialized output)
    """

    __test__ = False

    def get_name(self) -> str:
        return "loop_dyn_valid_one_block"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("sij_buf", [N_ROW, BLOCK_COL], DataType.FP32, init_value=torch.randn),
            TensorSpec("scale_cfg", [1], DataType.FP32, init_value=2.0),
            TensorSpec(
                "n_blocks_cfg",
                [1],
                DataType.INT64,
                init_value=torch.tensor([1], dtype=torch.int64),
            ),
            TensorSpec(
                "last_valid_len_cfg",
                [1],
                DataType.INT64,
                init_value=torch.tensor([48], dtype=torch.int64),
            ),
            TensorSpec(
                "block_size_cfg",
                [1],
                DataType.INT64,
                init_value=torch.tensor([64], dtype=torch.int64),
            ),
            TensorSpec("output", [N_ROW, BLOCK_COL], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        from examples.kernels.dyn_valid_shape import build_loop_program  # noqa: PLC0415

        return build_loop_program()

    def compute_expected(self, tensors: dict[str, torch.Tensor], _params=None) -> None:
        scale = float(tensors["scale_cfg"][0].item())
        data = tensors["sij_buf"].clone()
        # Output is zero-initialized; only block 0 is written
        expected = torch.zeros((128, 64), dtype=torch.float32)
        # Block 0 (also last): cols 0-47 valid, cols 48-63 = -inf
        expected[:64, :48] = data[:64, :48] * scale
        expected[:64, 48:] = float("-inf")
        tensors["output"][:] = expected


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoopDynValidShape:
    """Verify loop + if/else dynamic valid_shape produces correct results."""

    def test_two_blocks(self, test_runner):
        """2 blocks: block 0 full, block 1 partial (48 valid cols padded with -inf)."""
        result = test_runner.run(LoopDynValidTwoBlocksTestCase())
        assert result.passed, f"Test failed: {result.error}"

    def test_one_block(self, test_runner):
        """1 block: single block is the last → partial valid (48 cols), rest -inf."""
        result = test_runner.run(LoopDynValidOneBlockTestCase())
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
