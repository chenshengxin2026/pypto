# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Experimental WSE graph frontend and static A3 simulation compiler.

The graph is independent of physical topology, PG, routes and slot images.
This frontend deliberately supports only the static operations it can lower.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

FP16 = "f16"
FP32 = "f32"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def aligned_bytes(elements: int, dtype: str) -> int:
    """Return the size of a private local allocation in 2 KiB pages."""
    return ((elements * (2 if dtype == FP16 else 4) + 2047) // 2048) * 2048


@dataclass(frozen=True)
class Group:
    """Declare the number of logical workers in each collective group."""

    size: int

    def __post_init__(self) -> None:
        _require(type(self.size) is int and self.size > 0, f"Expected positive group size, got {self.size}")


@dataclass(frozen=True)
class Replicated:
    """Place one input copy in every worker's private local DRAM."""


@dataclass(frozen=True)
class ColumnSharded:
    """Partition columns contiguously, in logical worker order."""

    parts: int

    def __post_init__(self) -> None:
        _require(
            type(self.parts) is int and self.parts > 0, f"Expected positive shard count, got {self.parts}"
        )


@dataclass(frozen=True)
class Shard:
    begin: int
    end: int
    replica: int


def infer_groups(layout: tuple[Shard, ...], group: Group) -> list[list[int]]:
    """Infer ordered groups from feature intervals and replica provenance."""
    size = group.size
    _require(bool(layout) and len(layout) % size == 0, "Group size must divide logical parallelism")
    spans = {(s.begin, s.end) for s in layout}
    if len(spans) == len(layout):
        ordered = sorted(range(len(layout)), key=lambda i: layout[i].begin)
        groups = [ordered[start : start + size] for start in range(0, len(layout), size)]
    else:
        lanes = sorted({s.replica for s in layout})
        _require(len(lanes) == len(layout) // size, "Ambiguous replica layout for requested group size")
        groups = [
            sorted((i for i, s in enumerate(layout) if s.replica == lane), key=lambda i: layout[i].begin)
            for lane in lanes
        ]
    _require(sorted(i for g in groups for i in g) == list(range(len(layout))), "Incomplete logical partition")
    for members in groups:
        _require(len(members) == size, "Nonuniform group membership")
        shards = [layout[i] for i in members]
        _require(all(s.begin >= 0 and s.end > s.begin for s in shards), "Invalid feature interval")
        _require(len({s.end - s.begin for s in shards}) == 1, "Only equal-width AllGather is supported")
        _require(
            all(a.end == b.begin for a, b in zip(shards, shards[1:])), "Overlapping or nonadjacent shards"
        )
    return groups


@dataclass(frozen=True)
class Tensor:
    program: "Program"
    id: int
    shape: tuple[int, int]
    dtype: str
    layout: tuple[Shard, ...]

    def __mul__(self, other: "Tensor") -> "Tensor":
        _require(self.program is other.program, "Cannot combine tensors from different programs")
        _require(self.shape == other.shape and self.layout == other.layout, "Elementwise layouts must match")
        _require(self.dtype == other.dtype == FP32, "Multiplication requires FP32 inputs")
        return self.program._append("mul", [self, other], self.shape, FP32, self.layout)


class Program:
    """Build a logical tensor graph for a statically partitioned WSE program."""

    def __init__(self, *, workers: int = 32) -> None:
        _require(type(workers) is int and 1 <= workers <= 32, "Supported logical workers: 1..32")
        self.workers = workers
        self.nodes: list[dict] = []
        self.calls: list[dict] = []
        self.output: Tensor | None = None

    def _append(
        self,
        kind: str,
        inputs: list[Tensor],
        shape: tuple[int, int],
        dtype: str,
        layout: tuple[Shard, ...],
        **attrs: object,
    ) -> Tensor:
        tensor = Tensor(self, len(self.nodes), shape, dtype, layout)
        self.nodes.append(
            {
                "id": tensor.id,
                "kind": kind,
                "inputs": [x.id for x in inputs],
                "shape": list(shape),
                "dtype": dtype,
                "attrs": attrs,
                "layout": [[s.begin, s.end, s.replica] for s in layout],
            }
        )
        return tensor

    def input(
        self, name: str, shape: tuple[int, int], *, layout: Replicated | ColumnSharded, dtype: str = FP16
    ) -> Tensor:
        """Declare input data distribution, without exposing physical placement."""
        _require(len(shape) == 2 and all(type(d) is int and d > 0 for d in shape), f"Invalid shape: {shape}")
        _require(dtype == FP16, "WSE simulation inputs currently require FP16")
        _require(
            name and not any(n["attrs"].get("name") == name for n in self.nodes), "Duplicate/empty input name"
        )
        if isinstance(layout, ColumnSharded):
            _require(
                layout.parts == self.workers and shape[1] % layout.parts == 0,
                "Column partitions must match workers and divide columns",
            )
            width = shape[1] // self.workers
            spans = tuple(Shard(i * width, (i + 1) * width, 0) for i in range(self.workers))
            local_shape = (shape[0], width)
            placement = "column_sharded"
        else:
            _require(isinstance(layout, Replicated), "Unknown input layout")
            spans = tuple(Shard(0, shape[1], i) for i in range(self.workers))
            local_shape = shape
            placement = "replicated"
        return self._append(
            "input", [], local_shape, dtype, spans, name=name, global_shape=list(shape), placement=placement
        )

    def result(self, value: Tensor, *, axis: int = -1) -> Tensor:
        """Export column shards in logical order; this adds no collective."""
        _require(value.program is self and axis in (-1, 1), "Result must be a column-sharded tensor")
        ordered = sorted(value.layout, key=lambda s: s.begin)
        _require(
            ordered[0].begin == 0 and all(a.end == b.begin for a, b in zip(ordered, ordered[1:])),
            "Result shards must cover the feature axis exactly once",
        )
        self.output = value
        return value

    def emit(self, directory: str | Path, *, nodes: int = 48) -> Path:
        """Emit machine-instance-independent PTO IR, manifest and binding schema."""
        _require(self.output is not None, "Program has no sharded result")
        _require(nodes in (40, 48), "Supported target route profiles: 40 or 48 nodes")
        target = Path(directory).resolve()
        target.mkdir(parents=True, exist_ok=True)
        local_bytes = sum(aligned_bytes(n["shape"][0] * n["shape"][1], n["dtype"]) for n in self.nodes)
        _require(local_bytes <= 384 * 1024 * 1024, "Private local DRAM exceeds WSE 384 MiB budget")
        ub_bytes = max(
            [0]
            + [c["recvOffset"] + c["capacity"] for c in self.calls]
            + [
                n["shape"][0] * n["shape"][1] * (6 if n["kind"] == "cast" else 16)
                for n in self.nodes
                if n["kind"] in ("silu", "mul", "cast")
            ]
        )
        _require(ub_bytes <= 192 * 1024, "Local vector temporaries exceed A3 UB")
        uses_cube = any(n["kind"] == "gemm" for n in self.nodes)
        graph = {
            "schema": "pypto.wse.graph.v1",
            "workers": self.workers,
            "nodes": self.nodes,
            "output": self.output.id,
            "route_nodes": nodes,
        }
        manifest = {
            "schema": "wse.calendar.manifest.v1",
            "logicalWorkers": self.workers,
            "calls": self.calls,
            "localDramBytesPerWorker": local_bytes,
            "scratchHighWaterBytes": {
                "UB": ub_bytes,
                "L1": 6144 if uses_cube else 0,
                "L0A": 2048 if uses_cube else 0,
                "L0B": 2048 if uses_cube else 0,
                "L0C": 1024 if uses_cube else 0,
            },
            "localAllocations": [
                {"tensor": n["id"], "bytes": aligned_bytes(n["shape"][0] * n["shape"][1], n["dtype"])}
                for n in self.nodes
            ],
            "target": {
                "isa": "a3-mock",
                "nodes": nodes,
                "routeEntryBytes": 16,
                "routeABI": 2 if nodes == 40 else 3,
                "wseUBBytes": 384 * 1024,
                "a3UBBytes": 192 * 1024,
                "wseScratchLimits": {
                    "UB": 384 * 1024,
                    "L1": 1024 * 1024,
                    "L0A": 128 * 1024,
                    "L0B": 512 * 1024,
                    "L0C": 256 * 1024,
                },
                "localDramPageBytes": 2048,
            },
            "physicalTopology": None,
        }
        (target / "program.json").write_text(json.dumps(graph, indent=2) + "\n")
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (target / "binding_schema.json").write_text(
            json.dumps(
                {
                    "schema": "wse.bindings.v1",
                    "route_words": 2,
                    "mapping": "logical_to_physical",
                    "members": "[logical_destination][ordered_logical_sources]",
                    "control": ["binding_id", "plan_slot", "opcode", "installed", "epoch", "launch_id"],
                    "status": [
                        "error",
                        "committed_remote_bytes",
                        "epoch",
                        "binding_id",
                        "logical_id",
                        "physical_id",
                    ],
                    "statusStrideBytes": 64,
                    "buffers": "private node slots, 2048-byte aligned; only mock transport reads peers",
                },
                indent=2,
            )
            + "\n"
        )
        for node in self.nodes:
            if node["kind"] != "input":
                (target / f"op_{node['id']}.pto").write_text(_emit_kernel(node, self, nodes))
        return target

    def compile(self, directory: str | Path, *, ptoas: str | Path, nodes: int = 48) -> Path:
        """Verify native PTO IR and lower it through the modified PTOAS."""
        target = self.emit(directory, nodes=nodes)
        commands = []
        for node in self.nodes:
            if node["kind"] == "input":
                continue
            path = target / f"op_{node['id']}.pto"
            command = [str(ptoas), str(path), "--pto-arch=a3", "-o", str(path.with_suffix(".cpp"))]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            path.with_suffix(".log").write_text(result.stdout + result.stderr)
            commands.append({"command": command, "returncode": result.returncode})
            (target / "ptoas_commands.json").write_text(json.dumps(commands, indent=2) + "\n")
            if result.returncode:
                raise RuntimeError(f"PTOAS rejected {path.name}: {result.stderr[-3000:]}")
        return target


def matmul(lhs: Tensor, rhs: Tensor, *, acc_dtype: str = FP32) -> Tensor:
    """Multiply local matrices, accumulating into FP32."""
    _require(lhs.program is rhs.program and lhs.shape[1] == rhs.shape[0], "GEMM dimensions/program mismatch")
    _require(lhs.dtype == rhs.dtype == FP16 and acc_dtype == FP32, "Only FP16 inputs / FP32 accumulation")
    _require(
        all(s.begin == 0 and s.end == lhs.shape[1] for s in lhs.layout),
        "GEMM reduction dimension must be complete in every local worker",
    )
    m, k, n = lhs.shape[0], lhs.shape[1], rhs.shape[1]
    _require(
        1 <= m <= 16 and k % 64 == 0 and 64 <= k <= 65536 and n % 16 == 0 and n <= 4096,
        "A3 GEMM requires M<=16, K%64=0, K<=65536, N%16=0, N<=4096",
    )
    return lhs.program._append("gemm", [lhs, rhs], (m, n), FP32, rhs.layout, inner=k)


def silu(value: Tensor) -> Tensor:
    """Compute x / (1 + exp(-x)) without a clamp."""
    _require(value.dtype == FP32, "SiLU requires FP32 input")
    return value.program._append("silu", [value], value.shape, FP32, value.layout)


def cast(value: Tensor, dtype: str) -> Tensor:
    """Round FP32 to FP16 with round-to-nearest-even."""
    _require(value.dtype == FP32 and dtype == FP16, "Only FP32 -> FP16 cast is supported")
    return value.program._append("cast", [value], value.shape, dtype, value.layout)


def all_gather(value: Tensor, *, group: Group, axis: int = -1) -> Tensor:
    """Concatenate adjacent logical feature shards and retain replica identity."""
    _require(value.dtype == FP16 and axis in (-1, 1), "AllGather requires FP16 and the feature axis")
    members = infer_groups(value.layout, group)
    rows, cols = value.shape
    _require(1 <= rows <= 16 and cols % 16 == 0, "Calendar rows must be <=16 and source rows 32B aligned")
    output_layout: list[Shard | None] = [None] * value.program.workers
    replica_counts: dict[tuple[int, int], int] = {}
    for ids in members:
        interval = (value.layout[ids[0]].begin, value.layout[ids[-1]].end)
        for logical in ids:
            lane = replica_counts.get(interval, 0)
            output_layout[logical] = Shard(*interval, lane)
            replica_counts[interval] = lane + 1
    slot = len(value.program.calls)
    row_bytes = cols * 2
    recv_offset = max(8192, ((rows * row_bytes + 8191) // 8192) * 8192)
    capacity = rows * row_bytes * group.size
    _require(recv_offset + capacity <= 192 * 1024, "AllGather arena exceeds A3 UB")
    call = {
        "planSlot": slot,
        "groups": members,
        "groupSize": group.size,
        "groupCount": len(members),
        "opcode": 1,
        "redOp": 0,
        "dtype": FP16,
        "rows": rows,
        "rowBytes": row_bytes,
        "srcStride": row_bytes,
        "recvStride": row_bytes * group.size,
        "capacity": capacity,
        "expectedRxBytes": rows * row_bytes * (group.size - 1),
        "recvOffset": recv_offset,
        "selfOffsets": {str(logical): rank * row_bytes for g in members for rank, logical in enumerate(g)},
        "dependencies": [slot - 1] if slot else [],
        "globalDrainBefore": slot > 0,
    }
    value.program.calls.append(call)
    return value.program._append(
        "all_gather",
        [value],
        (rows, cols * group.size),
        FP16,
        tuple(s for s in output_layout if s is not None),
        plan_slot=slot,
    )


def _emit_kernel(node: dict, program: Program, nodes: int) -> str:
    idx, kind = node["id"], node["kind"]
    rows, cols = node["shape"]
    if kind == "all_gather":
        call = program.calls[node["attrs"]["plan_slot"]]
        types = ["!pto.ptr<f16>", "!pto.ptr<f16>", "!pto.ptr<i64>"] + ["!pto.ptr<i32>"] * 4
        attrs = {
            "rows": rows,
            "cols": call["rowBytes"] // 2,
            "group_size": call["groupSize"],
            "plan_slot": call["planSlot"],
            "recv_offset": call["recvOffset"],
            "workers": program.workers,
            "nodes": nodes,
            "opcode": 1,
            "red_op": 0,
        }
        op = "pto.tcalendar"
    else:
        dtype = program.nodes[node["inputs"][0]]["dtype"]
        types = [f"!pto.ptr<{dtype}>"] * 2 + [f"!pto.ptr<{node['dtype']}>", "!pto.ptr<i32>"]
        attrs = {
            "kind": kind,
            "rows": rows,
            "cols": cols,
            "inner": node["attrs"].get("inner", 0),
            "workers": program.workers,
        }
        op = "pto.wse_local"
    arguments = ", ".join(f"%arg{i}: {t}" for i, t in enumerate(types))
    operands = ", ".join(f"%arg{i}" for i in range(len(types)))
    attributes = ", ".join(
        f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v} : i64" for k, v in attrs.items()
    )
    role = "cube" if kind == "gemm" else "vector"
    return (
        f'module attributes {{pto.target_arch = "a3", pto.kernel_kind = #pto.kernel_kind<{role}>}} {{\n'
        f"  func.func @wse_op_{idx}({arguments}) attributes {{pto.kernel}} {{\n"
        f'    "{op}"({operands}) {{{attributes}}} : ({", ".join(types)}) -> ()\n'
        "    return\n  }\n}\n"
    )


__all__ = [
    "FP16",
    "FP32",
    "ColumnSharded",
    "Group",
    "Program",
    "Replicated",
    "Tensor",
    "all_gather",
    "cast",
    "matmul",
    "silu",
]
