# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Static WSE groups and synchronous Calendar collectives."""

from dataclasses import dataclass
from typing import overload

from pypto.language.typing import Tensor, Tile
from pypto.pypto_core import ir


@dataclass(frozen=True, eq=False)
class Group:
    """Ordered logical participants; declaration identity defines a communication domain."""

    block_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.block_ids) is not tuple or not self.block_ids:
            raise ValueError("Group.block_ids must be a nonempty tuple")
        if any(type(bid) is not int or not 0 <= bid < 2**31 for bid in self.block_ids):
            raise ValueError("Group members must be nonnegative int32 logical IDs")
        if len(set(self.block_ids)) != len(self.block_ids):
            raise ValueError("Group members must be distinct")

    @property
    def size(self) -> int:
        """Number of participants."""
        return len(self.block_ids)


@overload
def allgather(x: Tensor, *, group: Group, axis: int = 0) -> Tensor: ...


@overload
def allgather(x: Tile, *, group: Group, axis: int = 0) -> Tile: ...


def allgather(x: Tensor | Tile, *, group: Group, axis: int = 0) -> Tensor | Tile:
    """Gather equal static shards in group order, completing before returning.

    The A3 adapter currently accepts FP16 rank-2 inputs with 32-byte rows.
    Deployment supplies the logical identity, paired tables and shared epoch.

    Args:
        x: Local shard with positive static dimensions.
        group: Ordered logical participants in the collective.
        axis: Concatenation dimension; negative axes are normalized.

    Returns:
        A Tensor or Tile matching the input kind, concatenated in group order.

    Raises:
        TypeError: If group is not a Group.
        ValueError: If axis or the input geometry is unsupported.
    """
    if type(group) is not Group:
        raise TypeError("allgather requires a wse.Group")
    if type(axis) is not int or axis not in (-2, -1, 0, 1):
        raise ValueError("WSE allgather requires axis 0 or 1 for a rank-2 input")
    expr = x.unwrap()
    is_tile = isinstance(expr.type, ir.TileType)
    op = "tile.wse_allgather" if is_tile else "tensor.wse_allgather"
    call = ir.create_op_call(
        op,
        [expr],
        {"members": ",".join(str(bid) for bid in group.block_ids), "axis": axis % 2},
        ir.Span.unknown(),
    )
    return Tile(expr=call) if is_tile else Tensor(expr=call)


__all__ = ["Group", "allgather"]
