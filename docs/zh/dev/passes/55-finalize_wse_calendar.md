# FinalizeWseCalendar

在存储规划与 `MaterializeValidShapeSymbols` 之后、`LowerTileToBuffer` 之前，
确定同步 WSE Calendar 的最终契约。

每个独立内核只允许一次无条件 `tile.wse_allgather`。该 pass 拒绝循环或条件分支中的
集合通信，并要求 `MemoryPlanner.PYPTO`。输入和输出必须是已分配地址的行主序、
无分块（unboxed）L1 存储。操作的内存约束禁止发送和接收缓冲区别名；
PTOAS 随后校验具体地址范围不重叠及最终有效形状。

该 pass 添加 `exp_bytes = (members - 1) * rows * cols * sizeof(FP16)`。
代码生成必须取得此最终属性，并与 Tile 形状核对后才生成 `pto.tcalendar`。
本地分片占用接收容量，但不计入远端字节数。不含 Calendar 的程序保持原状。
遍历复杂度与 IR 大小成线性关系。

前端、部署和验证契约见 [A3 上的 WSE FFN](../wse-a3.md)。
C++ 入口为 `ir::pass::FinalizeWseCalendar()`，Python 入口为
`passes.finalize_wse_calendar()`。
