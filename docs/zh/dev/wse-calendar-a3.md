# WSE Calendar 的 A3 实验前端

`pypto.wse` 实现 1014 方案中的逻辑张量分布、分组和两轮 AllGather，并生成
修改版 PTOAS 可验证、可编译的原生 PTO IR。完整 FFN 位于
`examples/models/wse_calendar_ffn.py`。

这是独立于现有 `pypto.language` 解析器和 `pl.jit` pass pipeline 的实验前端。
设计伪码中的 `logical_parallel`、`dist.local_shard`、`dist.sharded_result`
分别落为 `Program(workers=...)`、`Program.input(layout=...)`、
`Program.result(...)`。现有 `pl.jit` 不会自动获得 WSE 能力。

## 编译与逻辑语义

- `Replicated()` 表示各逻辑 worker 的私有副本。
- `ColumnSharded(parts=32)` 按列连续切分。
- `Group(size=8)` 仅声明组内核数，不指定物理 mesh 方向。
- `all_gather(..., axis=-1)` 根据区间和副本来源推导成员序，第二轮选择不同
  区间的同一副本 lane，避免重复收集第一轮产生的相同副本。
- `result(...)` 按逻辑区间导出列分片，不隐式增加第三轮通信。

原始 FFN 为 `T=8,H=7168,I=3072,P=32`。本地中间张量依次为
`[8,96] → [8,768] → [8,3072] → [8,224]`。GEMM 使用 FP16 输入、FP32 累加，
SiLU 不做 clamp；与 up 分支相乘后将 Z 舍入为 FP16，再进入 down GEMM。

`Program.emit(directory, nodes=48)` 输出 `program.json`、`manifest.json`、
`binding_schema.json` 和每个 kernel 的 `op_*.pto`。manifest 保存逻辑组、接收
几何、调用依赖、内存预算；binding schema 定义部署参数及设备状态记录。
`Program.compile(..., ptoas=...)` 调用指定的原生 PTOAS，记录命令和诊断；
失败立即中止。`simpler.wse.build_device_library` 再用 bisheng 编译生成的 C++。

编译阶段不读取 PG、物理映射、路由或时隙表。`nodes` 是目标 ABI 选择，同一
ABI 下可用同一二进制部署不同映射与缺陷图。

## 支持范围与约束

支持静态二维 FP16 输入、FP32 GEMM/SiLU/乘法和 FP32→FP16 cast。
GEMM 要求 `M<=16`、`K%64=0且K<=65536`、本地 `N%16=0且N<=4096`。
部署 stub 当前支持 32 个 worker。AllGather 仅支持特征轴等宽分片，源行按
32 B 对齐；无法推导或有歧义的布局会报错。

local DRAM 按 2 KiB 页分配并限制为每 worker 384 MiB；A3 backing 地址也按
2 KiB 对齐。UB 收发区不得重叠；UB/L1/L0 用量同时受 WSE 与 A3 的较小容量约束。

默认 ABI 3 为白皮书的 6×8 拓扑提供 96 bit 路由；`nodes=40` 显式选择原
80 bit 的 ABI 2。两者保持 16 B 表项、偏移 12 的接收量。ABI 3 去掉 ABI 2
可由 bitmap 推导的 landing-count/flags 字段。这是仿真 ABI 决策，不能视为
对真实 WSE 拓扑待决问题的硬件确认。

当前不支持动态形状、通用切分、Reduce/AllReduce、原生 WSE 指令或 NoC 时序
仿真。A3 以隔离的 GM 槽位模拟私有 local DRAM，以真实 PTO DMA/向量指令
模拟 peer delivery；CPU golden 不参与设备结果生成。

## 验证

`tests/ut/wse/test_frontend.py` 覆盖原始几何、副本顺序、与机器实例无关的
输出、非法组、GEMM 形状和 UB 超限。联合验证还覆盖原生 PTOAS verifier、
部署拒绝路径、A3 FFN 精度、设备接收字节数，以及两轮 gather 的逐 bit 顺序。
