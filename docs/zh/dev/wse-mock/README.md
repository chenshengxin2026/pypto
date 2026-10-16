# WSE mock：PyPTO 前端

本分支新增实验性 `pypto.wse` 前端，用于在真实 A3 上执行 WSE Calendar FFN
仿真。上游基线为 `d06a0302f5b204acbc4102820b0a7e3edca1814d`。
PR 目标为 fork 的 `wse-mock`，待审代码位于 `wse-mock-calendar-a3`。

## 修改与接口

| 文件 | 作用 |
| ---- | ---- |
| [前端](../../../../python/pypto/wse/__init__.py) | 静态图、列分片归属、副本来源、资源校验、原生 PTO 输出 |
| [FFN 示例](../../../../examples/models/wse_calendar_ffn.py) | gate/up、SiLU/乘法、两轮 AllGather、down、分片输出 |
| [前端测试](../../../../tests/ut/wse/test_frontend.py) | shape、group、内存和代码输出的正反例 |

方案中的 `logical_parallel/local_shard/sharded_result` 分别落实为
`Program(workers)/Program.input(layout)/Program.result`。
`Group(size=8)` 和 `Group(size=4)` 描述逻辑分组，成员由副本来源推导。
该扩展独立于现有 `pl.jit` parser 和 pass pipeline。

`Program.compile(output, ptoas=..., nodes=48)` 输出图、manifest、binding schema、
PTO IR、原生 PTOAS 诊断和 C++。物理 PG、mapping、route、slot 在运行时绑定，
同一二进制可以重新部署而无需重编译。
[接口说明](../wse-calendar-a3.md) 列出了支持的形状和显式 40/48 节点 ABI。

## 构建与执行

配套使用
[PTOAS](https://github.com/chenshengxin2026/PTOAS/tree/wse-mock-calendar-a3)、
[pto-isa](https://github.com/chenshengxin2026/pto-isa/tree/wse-mock-calendar-a3)、
[simpler](https://github.com/chenshengxin2026/simpler/tree/wse-mock-calendar-a3)
的 `wse-mock-calendar-a3` 分支。PR 合入后可统一使用四仓 `wse-mock`。
先初始化本仓 submodule，再按根 README 构建当前 checkout。

本地原生 PyPTO 扩展构建完成后，在仓根目录执行：

```bash
export PYTHONPATH="$PWD/python:$PWD/examples/models${PYTHONPATH:+:$PYTHONPATH}"
source .claude/skills/testing/load-env.sh
python -m pytest -q --noconftest -c /dev/null -n "$PYPTO_TEST_JOBS" \
  tests/ut/wse/test_frontend.py
```

[跨仓运行指南](https://github.com/chenshengxin2026/simpler/blob/wse-mock-calendar-a3/docs/wse-mock/README.md)
提供可移植的 FFN 构建与 A3 golden 验证命令。需要显式指定修改版原生 PTOAS
可执行文件和 PTO-ISA checkout。

## 验证与边界

四仓实现于 2026-10-15 通过 46 项 CPU/PTOAS 定向测试、独立 C++ receiver
合约测试，以及 Ascend910_9392 上的 21 次 FFN 执行。
原始形状 `T=8,H=7168,I=3072,P=32`，覆盖两个 seed、三种部署、两次重复。
最终输出最大绝对误差 `0.015625`、最大相对 RMS `8.2343e-5`；
逐元素容差为 `atol=0.002, rtol=0.01`，相对 RMS 要求 `<0.002`。

两轮 gather 在每个 worker 上与设备输入分片逐 bit 一致；同 seed 下不同映射、
重复运行的最终输出逐 bit 一致。与 CPU golden 的比较使用上述容差。

当前支持 32 个逻辑 worker 的静态二维 FFN。PG/Batcher/CCE 使用 mock，
mapping 和 NoC 使用 stub。local DRAM 按 2 KiB 页分配，每 worker 上限
384 MiB；scratch 同时满足 WSE 和 A3 容量。验证覆盖功能与所建模约束，
不代表真实 WSE NoC 的时序或性能。
