"""五个维度的评测指标。

编排原则：**能规则化的绝不交给 LLM 判断。** 五个维度里四个完全或大部分
硬化了——任务完成看 `ending` 与 `known_fact_ids`，工具调用看 `ErrorCode`
枚举，助手腔直接用角色卡的禁语清单，行为一致性用 JS 散度对角色卡基线。

所有指标函数只读 `Trajectory` 与 `WorldDefs`：不跑引擎、不读 `WorldState`。
指标算错了可以拿旧轨迹重算，不用重跑模型。
"""

from gensokyo.testkit.metrics.hard import TaskMetrics, ToolMetrics, task_metrics, tool_metrics

__all__ = [
    "TaskMetrics",
    "ToolMetrics",
    "task_metrics",
    "tool_metrics",
]
