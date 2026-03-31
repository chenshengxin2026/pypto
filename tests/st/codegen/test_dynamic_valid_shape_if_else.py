# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Integration tests for dynamic valid_shape across if/else branches.

Verifies the PTO pattern where a tile buffer has dynamic valid shape and
the valid length is computed in an if/else:

  if is_last:
      vlen = last_valid_len   (partial block)
  else:
      vlen = full_len         (full block)
  tile = load(..., valid_shapes=[rows, vlen])
  padded = fillpad(tile, pad_value=PadValue.min)

Test scenarios:
  1. is_last=True  → valid_len=48 < 64: cols 48-63 padded with -inf, then scaled
  2. is_last=False → valid_len=64 = 64: no padding needed, then scaled
  3. Loop variant: iterate over 2 blocks, last block has reduced valid length
"""

from typing import Any

import pytest
import torch
from harness.core.harness import DataType, PTOTestCase, TensorSpec
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy

# ---------------------------------------------------------------------------
# Test case 1: is_last=True — partial valid_len, padding region filled with -inf
# ---------------------------------------------------------------------------


class DynValidShapeLastBlockTestCase(PTOTestCase):
    """Test: is_last=True, valid_len=48, full_len=64.

    Expected: cols 0-47 = input * scale, cols 48-63 = -inf (padded with min, then scaled).
    """

    __test__ = False

    def get_name(self) -> str:
        return "dyn_valid_shape_last_block"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("data", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("scale_cfg", [1], DataType.FP32, init_value=2.0),
            TensorSpec(
                "flag_cfg",
                [1],
                DataType.INT64,
                init_value=torch.tensor([1], dtype=torch.int64),
            ),
            TensorSpec(
                "valid_len_cfg",
                [1],
                DataType.INT64,
                init_value=torch.tensor([48], dtype=torch.int64),
            ),
            TensorSpec(
                "full_len_cfg",
                [1],
                DataType.INT64,
                init_value=torch.tensor([64], dtype=torch.int64),
            ),
            TensorSpec("output", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        from examples.kernels.dyn_valid_shape import build_if_else_program  # noqa: PLC0415

        return build_if_else_program()

    def compute_expected(self, tensors: dict[str, torch.Tensor], _params=None) -> None:
        scale = float(tensors["scale_cfg"][0].item())
        data = tensors["data"].clone()
        expected = torch.full((64, 64), float("-inf"), dtype=torch.float32)
        expected[:, :48] = data[:, :48] * scale
        # cols 48-63 remain -inf (pad.min * scale = -inf)
        tensors["output"][:] = expected


# ---------------------------------------------------------------------------
# Test case 2: is_last=False — full valid, fillpad is no-op
# ---------------------------------------------------------------------------


class DynValidShapeFullBlockTestCase(PTOTestCase):
    """Test: is_last=False, valid_len=48, full_len=64.

    Expected: all cols = input * scale (fillpad is no-op when valid == physical).
    """

    __test__ = False

    def get_name(self) -> str:
        return "dyn_valid_shape_full_block"

    def get_strategy(self) -> OptimizationStrategy:
        return OptimizationStrategy.Default

    def get_backend_type(self) -> BackendType:
        return BackendType.Ascend910B

    def define_tensors(self) -> list[TensorSpec]:
        return [
            TensorSpec("data", [64, 64], DataType.FP32, init_value=torch.randn),
            TensorSpec("scale_cfg", [1], DataType.FP32, init_value=2.0),
            TensorSpec(
                "flag_cfg",
                [1],
                DataType.INT64,
                init_value=torch.tensor([0], dtype=torch.int64),
            ),
            TensorSpec(
                "valid_len_cfg",
                [1],
                DataType.INT64,
                init_value=torch.tensor([48], dtype=torch.int64),
            ),
            TensorSpec(
                "full_len_cfg",
                [1],
                DataType.INT64,
                init_value=torch.tensor([64], dtype=torch.int64),
            ),
            TensorSpec("output", [64, 64], DataType.FP32, is_output=True),
        ]

    def get_program(self) -> Any:
        from examples.kernels.dyn_valid_shape import build_if_else_program  # noqa: PLC0415

        return build_if_else_program()

    def compute_expected(self, tensors: dict[str, torch.Tensor], _params=None) -> None:
        scale = float(tensors["scale_cfg"][0].item())
        data = tensors["data"].clone()
        tensors["output"][:] = data * scale


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDynValidShapeIfElse:
    """Verify dynamic valid_shape selection via if/else produces correct results."""

    def test_last_block(self, test_runner):
        """is_last=True: partial valid region, padding cols filled with -inf then scaled."""
        test_case = DynValidShapeLastBlockTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"

    def test_full_block(self, test_runner):
        """is_last=False: full valid region, fillpad is no-op, all cols scaled."""
        test_case = DynValidShapeFullBlockTestCase()
        result = test_runner.run(test_case)
        assert result.passed, f"Test failed: {result.error}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
