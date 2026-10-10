# 成员 B 工作交付记录

## 范围与源提交
- 分支：`grace/prompt-nki`。
- 提示词提交：`5dd32df9b946e24bb8452fe5142b1610174961a2`（Improve NKI prompts for levels 1 and 3）。
- 仅整理该提交和本次 L1/L3 验证；未修改 checker、评分规则、任务定义或队友文件，未运行全量测试、未合并 main。

## 提示词提交真实 diff
根据 `git show --numstat 5dd32df9b946e24bb8452fe5142b1610174961a2`：
| 文件 | 新增 | 删除 |
|---|---:|---:|
| `projects/02-kernel-agent/agent.py` | 11 | 1 |
| `projects/02-kernel-agent/kernel_guides.py` | 27 | 0 |
| `projects/02-kernel-agent/prompts.py` | 28 | 0 |
| **合计** | **66** | **1** |

- `prompts.py` 按 level 选择相关 NKI API 卡片，加入 Pmax 和 matmul free-dimension 限制；按 level 载入知识，不把所有 API 堆进每个提示。
- `kernel_guides.py` 的 L1 指引具体说明输入/输出形状、pool 窗口两轴归约及除以窗口面积、DMA 源/目标元素数匹配、partition 限制和边界自查。
- L3 指引具体说明 lhsT=(K,M)、rhs=(K,N)、输出=(M,N)，二维 SBUF/PSUM tile、HBM→SBUF→PSUM→SBUF→HBM 数据流；本关 K=128、M=64、N=512 可单 tile 完成。
- `agent.py` 在原接口 `first_prompt(level, terse=0)` 中接入 level context，普通和两种 terse 提示路径都保留 context。输出仍要求单个 Python 代码块。删除的一行是通用 API 卡片插入点，由分级 context 替换。

## 本次 L1/L3 验证
seat-22 app 容器，HEAD 为 `5dd32df9b946e24bb8452fe5142b1610174961a2`；Qwen3-8B 服务响应正常。每级单次运行参数：8 rounds、4 samples、context 8192、repeat 1。

| Level | rounds | 尝试记录 | Reward / 解决 | 正确性与执行 |
|---|---:|---:|---|---|
| L1 | 8/8 | 32 | 0.30；未解决（0/1） | 0/4 shapes 通过；解析和静态规则得分，NKI 模拟执行失败，正确性未通过。 |
| L3 | 5/8（重复错误提前停止） | 20 | 0.30；未解决（0/1） | 0/1 shape 通过；解析和静态规则得分，NKI 模拟执行失败，正确性未通过。 |

L1 代表性错误：`nl.Layout`、`nl.OpCode` 不存在；`ndarray(layout=...)` 不支持；`tensor_scalar` 的 `dtype=` 不支持且调用缺少必需参数。L3 代表性错误：`dma_copy(src_layout=...)` 不支持；之后反复把 SBUF 传作 `nc_matmul` 输出，但输出必须位于 PSUM；同一错误连续四轮后 agent 停止。

该项目当前通过 CPU 上的 `nki.simulate` 做检查，硬件延迟与 profile 测量层尚未实现。本次没有 kernel 性能数据；生成耗时不作为 kernel 性能。

## Baseline 与结论限制
仓库 `STATE.md` / 项目说明记录的 Qwen3-8B baseline 为 5 次重复运行，每次 8 rounds × 4 samples、context 8192。L1 和 L3 均为 0/5 解决、每次 Reward 0.30。本次每级只运行 1 次，单次配置相同但重复次数少于 baseline。首轮 L1/L3 的 Reward 与 baseline 记录一致；该轮没有观察到变化，也不能证明稳定性变化。

## 附带验证日志
- `projects/02-kernel-agent/validation-runs/5dd32df-20261010/l1-console.log`
- `projects/02-kernel-agent/validation-runs/5dd32df-20261010/l1-attempts.jsonl`
- `projects/02-kernel-agent/validation-runs/5dd32df-20261010/l3-console.log`
- `projects/02-kernel-agent/validation-runs/5dd32df-20261010/l3-attempts.jsonl`

## B 任务第二轮：L1 API 修正与验证

### API 核对及改动

- seat-22 app 中安装的 NKI 为 `0.6.0+31049202112.g85070674`。实测签名：`nl.ndarray(shape, dtype, buffer=..., name='', address=None)`；`nisa.tensor_scalar(dst, data, op0, operand0, reverse0=False, op1=None, operand1=None, reverse1=False, engine=..., name=None)`；`nl.sum(x, axis, dtype=None, keepdims=False)`。安装版没有 `nl.Layout`、`nl.OpCode`；有 `nl.multiply`。
- 对照 `projects/02-kernel-agent/reference_level1.py` 和安装源 `nki/language/tensor.py`：pool view 用 `in_tile.ap(pattern)`，pattern 是 `[stride,count]` 对列表；通过 `nl.sum(pool_view, axis=[3,4])` 求和，再使用 `nisa.tensor_scalar(dst=scaled, data=sums, op0=nl.multiply, operand0=1.0/(p*p))` 归一化。
- 本轮只改 `prompts.py` 的 L1 API 卡和 `kernel_guides.py` 的 L1 guide：加入精确参数、禁用未存在的类型/错误 keyword，并加入可复制的最小 pooling 数据流示例。相对上一提交，guide 为 +17/-6 行，prompt API 卡为 +1/-1 行。现有 `first_prompt(level, terse=0)` 已接入 context，故未改 `agent.py`。
- Python 语法检查及 `first_prompt(1)` 内容接入检查通过。

### 执行验证和上一轮差异

1. `python nkibench.py --level 1 --check reference_level1.py`：rules clean，4/4 shapes 数值通过。checker 报告 HBM 流量为理论下限 1.00x；这是流量估算，不是设备延迟实测。
2. 单次小规模 L1：`--rounds 1 --samples 1 --context 8192 --repeat 1`；round 0 在 18.2 秒生成 1 个候选，Reward 1.00，全部 shapes 正确，1/1 solved。生成代码采用核实过的参考 API 和 pooling 数据流。
3. 上一轮 L1 为 8 rounds × 4 samples、32 条尝试，Reward 0.30、0/4 shapes 通过。本轮的 1 sample/1 round 与上一轮以及 baseline 的 5 次重复配置不具备相同样本规模。baseline 记录为 L1 0/5 solved、Reward 0.30；本次单个成功样本是积极观测，但不足以证明稳定、可重复提分。**尚未证明整体提分。**

本轮新增日志：

- `projects/02-kernel-agent/validation-runs/5dd32df-20261010/l1-api-fix-console.log`
- `projects/02-kernel-agent/validation-runs/5dd32df-20261010/l1-api-fix-attempts.jsonl`
