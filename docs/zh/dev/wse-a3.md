# A3 上的 WSE Calendar FFN

`wse-sim-1026` 分支实现了 WSE Calendar 设计的同步可执行子集。
[FFN 用例](../../../examples/models/wse_ffn.py) 使用常规 PyPTO GEMM、向量激活、
类型转换，以及 `pl.wse.allgather(x, group=pl.wse.Group((2, 5)), axis=1)`。

四个有序逻辑成员 `(2, 5, 9, 12)` 分别持有列分片。B 组为 `(2, 5)` 和 `(9, 12)`，
C 组为 `(2, 9)` 和 `(5, 12)`。用例取 `T=16, H=64, I=128`：每个成员计算
`Z = FP16(SiLU(X @ W_gate) * (X @ W_up))` 的 32 列，两轮集合通信分别得到
64、128 列，再计算 `FP16(Z @ W_down)` 的 16 列。矩阵乘法采用 FP32 累加。
输出保留四个分片，没有第三轮集合通信。

## 下载四个 fork 并复现

需要 Linux AArch64、A3、CANN 9.0（PATH 中可找到 `bisheng`）、兼容的
`torch`/`torch_npu`、Python 3.10+、C++17 编译器、CMake 和 Ninja。
PTOAS 需要 `vpto-dev/llvm-project` 的 `feature-vpto` 分支 LLVM19/MLIR，
启用 Python bindings，并使用 `pybind11<3`；工具链的准备方法见 PTOAS README。
本次验证的 LLVM 源码版本为 `fa2fd1f75d742fec91ecf40d98c9c7961647ec42`。
将 `LLVM_BUILD_DIR` 指向其构建目录。以下所有设备命令均经 `task-submit` 执行。

```bash
mkdir wse-sim && cd wse-sim
for repo in pypto simpler PTOAS pto-isa; do
  git clone --branch wse-sim-1026 --single-branch \
    "https://github.com/chenshengxin2026/$repo.git" "$repo"
done
git -C pypto submodule update --init 3rdparty/libbacktrace 3rdparty/msgpack-c
python3 -m venv --system-site-packages pypto/.venv
source pypto/.venv/bin/activate
python -m pip install 'nanobind>=2,<3' 'pybind11<3' cmake ninja numpy pytest

cmake -S pypto -B pypto/build -DCMAKE_BUILD_TYPE=Release \
  -Dnanobind_DIR="$(python -m nanobind --cmake_dir)"
cmake --build pypto/build --parallel 2
cmake -S PTOAS -B PTOAS/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPTO_ENABLE_PYTHON_BINDING=ON -DBUILD_TESTING=OFF -DENABLE_PACKAGE=OFF \
  -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
  -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build PTOAS/build --parallel 2 --target PTOASPythonPackage
bash pypto/examples/models/run_wse_a3.sh "$(command -v python)"
```

PTOAS 会获取 `master-055` 版本的 `cann-cmake`。离线环境可传入
`-DCANN_3RD_LIB_PATH=/path/to/deps`，其中包含 `cann-cmake/`；本次使用的提交为
`a7ef9a7f04ea9efc5ad388cdb8cce0b0c1759c70`，复现时应核对依赖版本。
此适配器直接使用同级 simpler 仓的实际 Tensor ABI，不要求构建 simpler 的原生扩展，
也不要求初始化 PyPTO 的 runtime 子模块。该子模块已固定到配套 fork 提交，
相对 URL 会跟随 PyPTO fork 的所有者。下载源码 ZIP 时保持上述目录布局，
并单独补齐 PyPTO 固定版本的两个第三方依赖目录。

脚本在 `pypto/build_output/wse_ffn/` 下保存各编译 pass 的 IR、PTO IR、PTOAS C++、
原始 PyPTO `kernel_entry` 包装函数、AIC/AIV 动态库、编译签名、编译命令和日志、
设备日志及 `validation.json`。三个随机种子各运行正向、反向两种物理启动顺序；
任何失败均以非零状态退出。

## 指令下沉和执行契约

| 层 | 职责 |
| --- | --- |
| PyPTO | JIT/parser 保留有序逻辑成员；Tensor 下沉为平铺 L1 Tile；使用 `MemoryPlanner.PYPTO` 分配不重叠地址；分配后确定远端字节数；生成 `pto.tcalendar` 和隐藏上下文参数 |
| PTOAS | 校验类型、最终有效形状、期望字节数和静态 L1 地址；保留通信副作用；生成 `pto::wse::RunCalendar<axis, members...>` |
| PTO-ISA | `TCalendar<Plan>` 依次完成依赖、BSP、接收布防（arm）、发送、远端有效载荷提交和本地复制，同步返回 |
| CCE A3 mock | 真实 cbuf/L1 端点与 MTE DMA；通过每次启动独享的 GM mailbox 模拟传输；以 epoch 和逐来源计数校验完成条件 |
| simpler | 以黑盒桩（stub）生成 mapping 及配对路由/时隙表，安装表、分配共享 epoch，将逻辑成员与物理启动顺序解耦，启动驻留组并检查状态 |

`exp_val = (group.size - 1) * rows * cols * 2` 仅统计远端有效字节，接收地址容量
则包含本地分片。接收指令只布防，不阻塞；每个核都能先发出 SEND，再等待载荷。
同步门面在返回及释放 L1 存储前完成所有 DMA。编译器不会删除结果未使用的集合通信。

每个组有独立部署和 BSP 句柄。表在部署时安装一次；每次调用独享 epoch、mailbox、
信号及状态，直到 stream 完成。配对 stub 表表达完整组播受众和确定性时隙，
不模拟 NoC 时序和仲裁。

A3 没有原生 Calendar 指令，也不保证跨这些启动保留 L1。因此适配器用真实 GM DMA
模拟传输，并在 A/B/C/D 内核之间显式写回 GM。B/C 各组依次启动，同一组的成员同时驻留。
该实现验证真实 A3 上的编译链、通信语义和 FFN 精度，不提供原生 WSE 吞吐量、
路由可行性或 32 成员驻留的证明。

支持：静态二维 FP16 AllGather、轴 0/1（前端接受负轴）、源行 32 字节对齐、
1–24 个成员、接收容量不超过 128 KiB、每个独立内核一次无条件集合通信、A3、
PYPTO 内存规划。异步通信、归约、动态或尾块形状、单内核多次 Calendar、
分布式窗口操作数、PTOAS 内存规划、普通 ChipWorker 编排绑定均不在当前范围内。
请编译独立的 `@pl.jit.incore` 内核并用 `simpler.wse` 执行。

## 验证

通信结果与实际设备产生的输入分片拼接结果逐位比较，与 FFN 数值基准独立。
完整 FFN 基准采用 CPU FP32 矩阵乘法/SiLU，并在对应位置做 FP16 舍入。
测试检查每轮通信、逻辑顺序、epoch、状态和字节数；最终精度阈值为
`rtol=0.002, atol=0.0001`。轴 0、稀疏乱序逻辑 ID、单成员零远端字节契约也在板上执行。

```bash
PYTHONPATH=pypto/python python -m pytest pypto/tests/ut/language/test_wse.py -q
python PTOAS/test/wse_calendar.py --ptoas PTOAS/build/tools/ptoas/ptoas
g++ -std=c++17 -Wall -Wextra -Werror -Ipto-isa/include \
  pto-isa/tests/cpu/wse/calendar_contract.cpp -o /tmp/wse-contract
/tmp/wse-contract
PYTHONPATH=simpler/python python -m pytest \
  --confcutdir=simpler/tests/ut/py/wse simpler/tests/ut/py/wse -q
```

验证组织方式参考 [pto-isa PR #252](https://github.com/hw-native-sys/pto-isa/pull/252)
的分层契约、传输与数值检查风格；指令 ABI 和 FFN 数据分布遵循 Calendar 设计。

[本次 A3 验证记录](../../wse-a3-validation.json) 包含 Ascend910_9392 上通过的
六次运行（种子 1026–1028、两种映射）。FFN 最大绝对误差为 `3.0517578125e-5`，
所有 AllGather 均逐位一致。B/C 每成员远端提交量为 1024/2048 字节，
轴 0 为 64 字节，单成员为 0。记录包含任务 ID、设备/工具版本、epoch 和产物哈希。

WSE 专项检查通过，但本机未取得全仓单测全绿。附加回归受到 Release 构建缺少
调试回溯、普通 ChipWorker 测试加载旧版系统 `_task_interface` 原生扩展，以及
CANN 驱动信息污染一项子进程标准输出检查的影响。JSON 记录分别列出这些限制。
运行这些普通 runtime 单测需要重建匹配版本的 simpler 原生扩展，并使用
Debug/RelWithDebInfo 编译器构建；WSE 适配器不依赖该扩展。
