# Checker Feedback：成员 C 接入说明

成员 A 已将 Checker 的自然语言报错转换为稳定、可序列化的结构化反馈。评分权重、正确性判断、测试形状和通过条件均未改变。

## C 可以直接使用的入口

```python
from agent import grade_detailed

reward, parts, feedback = grade_detailed(candidate_source, level)

payload = feedback.to_dict()   # 写日志、做路由、统计错误类型
prompt = feedback.to_prompt()  # 直接交给下一轮代码生成 Agent
breakdown = feedback.reward_breakdown  # 每个评分项的得分、上限和 shape 通过数
```

现有 `grade(source, level)` 仍返回 `(reward, parts, feedback_text)`，旧调用方无需修改。

Agent 运行日志中的每条记录现在同时包含：

```json
{
  "feedback": "给生成模型使用的可执行建议",
  "feedback_structured": {
    "schema_version": "1.1",
    "category": "PARTITION_LIMIT",
    "family": "TILE_LAYOUT",
    "severity": "structural",
    "stage": "rules",
    "summary": "Partition dimension 256 exceeds the hardware maximum 128",
    "suggestion": "Tile that partition dimension in chunks of at most 128; keep the final chunk partial",
    "redesign_required": true,
    "retryable": true,
    "raw_error": "dma_copy dst partition dimension 256 exceeds maximum 128",
    "reward": 0.5,
    "reward_breakdown": {
      "parses": {"passed": true, "earned": 0.1, "maximum": 0.1},
      "rules": {"passed": true, "earned": 0.2, "maximum": 0.2},
      "runs": {"passed": true, "earned": 0.2, "maximum": 0.2},
      "correct": {
        "passed": false,
        "earned": 0.0,
        "maximum": 0.5,
        "passed_cases": 0,
        "total_cases": 3
      }
    },
    "current": 256,
    "maximum": 128,
    "dimension": "partition",
    "expected": null,
    "actual": null,
    "level": 4,
    "case": null,
    "shape": null
  }
}
```

## 推荐路由规则

- `retryable == false`：不要再次调用代码生成 Agent；这是环境问题或已经成功。
- `category == VERIFIED`：结束当前任务。
- `redesign_required == true`：允许 C 要求模型调整 tile、循环或内存布局。
- 其他可重试错误：要求模型只修复 `suggestion` 指出的局部问题。
- `UNKNOWN_ERROR`：显示 `raw_error`，记录新样本；不要静默吞掉。

`category` 是供程序判断的稳定标识；不要解析 `summary` 或 `suggestion` 的自然语言。

`reward_breakdown` 中的 `earned` 相加始终等于顶层 `reward`。其中 `correct`
支持按通过 shape 数量给部分分，因此同时提供 `passed_cases` 和 `total_cases`。

## 当前分类族

- `FORMAT`：空回复、Python 语法或导入失败。
- `RULE`：入口函数、装饰器或比赛规则违规。
- `API`：不存在的 NKI API、参数或内存区域误用。
- `TILE_LAYOUT`：partition、shape、越界、matmul contraction、buffer placement。
- `RUNTIME`：加载失败、模拟异常、硬件风险。
- `NUMERICAL`：输出 shape、NaN/Inf、数值误差、ragged edge、输入被修改。
- `PERFORMANCE`：HBM traffic 超标。
- `ENVIRONMENT`：Neuron SDK/模拟环境不可用。
- `SUCCESS`：全部检查通过。

## 验证命令

```bash
python -m unittest -v test_feedback.py
python nkibench.py --selftest
```

第一条验证反馈结构和兼容接口；第二条验证原 Checker 的规则、参考实现、数值诊断和 roofline 逻辑没有发生回归。
