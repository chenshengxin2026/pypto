# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Four logical workers, two ordered AllGather stages, and a sharded SwiGLU FFN.

Compile with this checkout and PTOAS; run with simpler.wse on A3. Kernel
boundaries spill to local GM because A3 does not retain L1 across launches.
See docs/en/dev/wse-a3.md for the reproducible four-repository workflow.
"""

import argparse
import json
from pathlib import Path

import pypto.language as pl
from pypto import ir, passes

MEMBERS = (2, 5, 9, 12)
B_GROUPS = (pl.wse.Group((2, 5)), pl.wse.Group((9, 12)))
C_GROUPS = (pl.wse.Group((2, 9)), pl.wse.Group((5, 12)))
T, H, INTERMEDIATE = 16, 64, 128


def make_gather(rows: int, cols: int, group: pl.wse.Group, axis: int = 1):
    out_rows = rows * group.size if axis == 0 else rows
    out_cols = cols * group.size if axis == 1 else cols

    @pl.jit.incore
    def gather(
        x: pl.Tensor[[rows, cols], pl.FP16],
        output: pl.Out[pl.Tensor[[out_rows, out_cols], pl.FP16]],
    ):
        result = pl.wse.allgather(x, group=group, axis=axis)
        output = pl.assemble(output, result, [0, 0])
        return output

    return gather


def make_matmul(m: int, k: int, n: int):
    @pl.jit.incore
    def matmul(
        x: pl.Tensor[[m, k], pl.FP16],
        weight: pl.Tensor[[k, n], pl.FP16],
        output: pl.Out[pl.Tensor[[m, n], pl.FP32]],
    ):
        a = pl.load(x, [0, 0], [m, k], target_memory=pl.MemorySpace.Mat)
        b = pl.load(weight, [0, 0], [k, n], target_memory=pl.MemorySpace.Mat)
        left = pl.move(a, target_memory=pl.MemorySpace.Left)
        right = pl.move(b, target_memory=pl.MemorySpace.Right)
        product = pl.matmul(left, right)
        pl.store(product, [0, 0], output)
        return output

    return matmul


@pl.jit.incore
def swiglu(
    gate: pl.Tensor[[T, INTERMEDIATE // 4], pl.FP32],
    up: pl.Tensor[[T, INTERMEDIATE // 4], pl.FP32],
    output: pl.Out[pl.Tensor[[T, INTERMEDIATE // 4], pl.FP16]],
):
    g = pl.load(gate, [0, 0], [T, INTERMEDIATE // 4])
    u = pl.load(up, [0, 0], [T, INTERMEDIATE // 4])
    neg = pl.mul(g, -1.0)
    denominator = pl.add(pl.exp(neg), 1.0)
    result = pl.mul(pl.mul(g, pl.recip(denominator)), u)
    rounded = pl.cast(result, pl.FP16)
    pl.store(rounded, [0, 0], output)
    return output


@pl.jit.incore
def output_cast(
    x: pl.Tensor[[T, H // 4], pl.FP32],
    output: pl.Out[pl.Tensor[[T, H // 4], pl.FP16]],
):
    value = pl.load(x, [0, 0], [T, H // 4])
    rounded = pl.cast(value, pl.FP16)
    pl.store(rounded, [0, 0], output)
    return output


def compile_example(output: Path, *, skip_ptoas: bool = False):
    kernels = {
        "gate_up": make_matmul(T, H, INTERMEDIATE // 4),
        "swiglu": swiglu,
        "down": make_matmul(T, INTERMEDIATE, H // 4),
        "output_cast": output_cast,
    }
    calendars = {}
    for stage, groups, cols in (("b", B_GROUPS, INTERMEDIATE // 4), ("c", C_GROUPS, INTERMEDIATE // 2)):
        for index, group in enumerate(groups):
            name = f"{stage}{index}"
            kernels[name] = make_gather(T, cols, group)
            calendars[name] = {"members": group.block_ids, "shape": [T, cols], "axis": 1}
    # Exercise both axes and the zero-remote-byte singleton contract.
    for name, group, axis in (("axis0", pl.wse.Group((9, 2)), 0), ("singleton", pl.wse.Group((17,)), 1)):
        kernels[name] = make_gather(2, 16, group, axis)
        calendars[name] = {"members": group.block_ids, "shape": [2, 16], "axis": axis}
    manifest = {"shape": [T, H, INTERMEDIATE], "members": MEMBERS, "calendars": calendars, "kernels": {}}
    for name, kernel in kernels.items():
        directory = output / name
        ir.compile(
            kernel.specialize(),
            output_dir=str(directory),
            memory_planner=passes.MemoryPlanner.PYPTO,
            runtime=passes.RuntimeKind.HOST_BUILD_GRAPH,
            platform="a2a3",
            dump_passes=True,
            skip_ptoas=skip_ptoas,
        )
        sources = list(directory.glob("kernels/aic/*.cpp")) + list(directory.glob("kernels/aiv/*.cpp"))
        if not skip_ptoas:
            if len(sources) != 1:
                raise RuntimeError(f"Expected one standalone kernel for {name}, got {sources}")
            manifest["kernels"][name] = str(sources[0].relative_to(output))
    output.mkdir(parents=True, exist_ok=True)
    (output / "wse_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build_output/wse_ffn"))
    args = parser.parse_args()
    compile_example(args.output)
