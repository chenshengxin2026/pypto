# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json

import pytest
from pypto import wse
from wse_calendar_ffn import build_ffn


def test_ffn_provenance_and_geometry():
    p = build_ffn()
    b, c = p.calls
    assert b["groups"] == [list(range(r * 8, (r + 1) * 8)) for r in range(4)]
    assert c["groups"] == [[r * 8 + lane for r in range(4)] for lane in range(8)]
    assert [(x["capacity"], x["expectedRxBytes"], x["recvOffset"]) for x in p.calls] == [
        (12288, 10752, 8192),
        (49152, 36864, 16384),
    ]
    assert p.output.shape == (8, 224)
    assert len(p.calls) == 2


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_bad_group_size(size):
    with pytest.raises(ValueError):
        wse.Group(size)


def test_groups_follow_provenance_not_ids():
    layout = tuple(wse.Shard(i * 16, (i + 1) * 16, 0) for i in reversed(range(32)))
    assert wse.infer_groups(layout, wse.Group(8))[0] == list(range(31, 23, -1))


def test_ambiguous_replica_rejected():
    layout = tuple(wse.Shard(0, 16, 0) for _ in range(32))
    with pytest.raises(ValueError, match="Ambiguous"):
        wse.infer_groups(layout, wse.Group(4))


def test_nonadjacent_or_nondivisible_rejected():
    layout = tuple(wse.Shard(i * 32, i * 32 + 16, 0) for i in range(32))
    with pytest.raises(ValueError, match="nonadjacent"):
        wse.infer_groups(layout, wse.Group(8))
    with pytest.raises(ValueError, match="divide"):
        wse.infer_groups(layout, wse.Group(3))


def test_unsupported_gemm_shape():
    with pytest.raises(ValueError, match="GEMM requires"):
        build_ffn(8, 64, 128)


def test_emit_is_instance_independent(tmp_path):
    p = build_ffn()
    a, b = p.emit(tmp_path / "a"), p.emit(tmp_path / "b")
    assert (a / "manifest.json").read_bytes() == (b / "manifest.json").read_bytes()
    assert (a / "op_9.pto").read_text().count('"pto.tcalendar"') == 1
    assert "physicalTopology" in (a / "manifest.json").read_text()
    assert "routeBits" not in (a / "op_9.pto").read_text()
    manifest = json.loads((a / "manifest.json").read_text())
    assert manifest["localDramBytesPerWorker"] == 4335616
    assert manifest["scratchHighWaterBytes"] == {
        "UB": 65536,
        "L1": 6144,
        "L0A": 2048,
        "L0B": 2048,
        "L0C": 1024,
    }
    assert all(entry["bytes"] % 2048 == 0 for entry in manifest["localAllocations"])


def test_ub_capacity_rejected():
    p = wse.Program()
    x = p.input("x", (16, 131072), layout=wse.ColumnSharded(32))
    with pytest.raises(ValueError, match="UB"):
        wse.all_gather(x, group=wse.Group(8))


def test_cross_program_tensor_rejected():
    a, b = build_ffn(), build_ffn()
    with pytest.raises(ValueError):
        wse.matmul(a.output, b.output)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
