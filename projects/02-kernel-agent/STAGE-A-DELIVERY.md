# Kernel Agent Challenge — Stage A 交付与复现

本目录同时包含两条路线。Challenge Stage A 使用 `stage_a_agent.py + kernelbench.py`，生成并验证普通 NumPy Kernel；原 Project 2 的真实 NKI 路线使用 `agent.py + nkibench.py`。两套 Level 编号含义不同，本交付只报告 Challenge Stage A 的十级算子结果。

## 交付物

| Challenge 要求 | 文件/输出 |
|---|---|
| Agent | `stage_a_agent.py` |
| Verification harness | `kernelbench.py` |
| Eval set | `kernelbench.py` 中每级的 `build` cases |
| Failure taxonomy | `stage_a_report.json` 的 `failure_taxonomy` |
| Token instrumentation | JSONL 每次尝试的 `token_budget`，以及汇总报告的 `token_instrumentation` |
| Reproduction note | 本文件 |

## 一键复现

只需要 Python 和 NumPy，不需要 AWS、Kubernetes、Neuron SDK 或 Trainium：

```bash
cd projects/02-kernel-agent

# 1. 证明 Checker 能拒绝作弊，并能抓住十个针对性的真实错误。
python kernelbench.py --selftest

# 2. 运行全部回归测试：正确实现必须通过，错误实现必须被稳定分类。
python -m unittest -v test_stage_a.py

# 3. 确定性地演示“错误候选 → 可执行反馈 → 正确候选”的完整闭环。
python stage_a_agent.py --offline --all --rounds 2 \
  --log stage_a_attempts.jsonl --report stage_a_report.json
```

离线模式第一轮固定重放每级的植入错误，第二轮重放规则合规的正确实现，用于验证 Agent/Checker 接线和报告生成。它不是模型能力实验，报告中会明确写入 `offline_smoke_test`，不得把其 10/10 当成模型 solve rate。

连接 OpenAI-compatible 模型后进行真实实验：

```bash
python stage_a_agent.py --all --rounds 6 --repeat 5 \
  --base http://localhost:8000/v1 --model Qwen/Qwen3-8B --no-think \
  --context 8192 --log model_attempts.jsonl --report model_report.json
```

报告成功率而不是最好的一次。每次尝试会保存候选代码、原始 verdict、失败类别、下一步修改指令、基于测试覆盖率的置信度、prompt/completion token，以及输入 token 在 reference、candidate、feedback、failure ledger 和 scaffolding 之间的估算分配。

## Checker 标准

- Tile 上限：128 行 × 512 列；静态检查会拒绝常见的超大 `range`/slice。
- 禁止把整个运算交给 NumPy：按 Level 拒绝 `sum`、`max`、`transpose`、`matmul`、`@`、`.T` 等捷径，并要求显式循环。
- 数值标准：`rtol=1e-4`、`atol=1e-5`。候选主要输出 float32，而参考实现用 float64 累加后转回目标类型；相对容差覆盖正常尺度，绝对容差避免接近零的正确结果被无意义的相对误差放大。
- Eval shapes 包含非整除 Tile、质数维度、维度 1、非方阵和 K 尾块。
- Hostile values 明确包含零、严格负数、常量行、大数，以及“大均值 + 小方差”。Attention 包含 `w > sequence`，Conv1D 包含 stride/dilation 尾部情形。

## 十级错误分类

1. ragged edge；2. reduction 尾块；3. max identity；4. accumulator precision；5. softmax stability；6. transpose ragged edge；7. matmul K tail；8. layernorm cancellation；9. attention window；10. convolution geometry。

`kernelbench.py --selftest` 必须证明十个植入错误全部被抓住，并且反馈中含有足以执行下一步修改的具体信息。`test_stage_a.py` 同时证明全部十个正确实现通过相同规则和数值测试，防止 Checker 只会拒绝错误代码却误伤正确代码。

## 当前边界

Stage A 已完成并可在 CPU 上复现。真实 NKI/Trainium 是 Challenge Stage B 的扩展路线；它不应与本目录中原有 Project 2 的 NKI Level 编号或结果混报。在线模型的 solve rate 必须在实际端点运行后从 `model_report.json` 如实报告，不能用确定性离线烟雾测试替代。
