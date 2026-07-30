# 墟镇 · 工程框架设计

**配套文档**：`2026-07-30-hollowtown-npc-design.md`（设计与玩法）
**本文范围**：代码结构、模块契约、数据流、配置格式、技术栈、测试与仓库工程化
**日期**：2026-07-30

---

## 0. 仓库定位

仓库将公开在 `github.com/Rohawku`。这意味着两件事必须在框架层面就考虑，不能事后补：

1. **仓库是门面。** 面试官点进 GitHub 主页，第一眼看到的是 README 和目录结构，不是代码细节。README 必须在 30 秒内讲清「这是什么、长什么样、怎么跑起来」。
2. **代码要经得起读。** 开源代码的第一读者是别人。模块边界清晰、依赖方向单一、能不看内部就知道一个模块干什么——这些从可有可无变成必须。

**建议命名**

| 项 | 值 |
|---|---|
| 仓库名 | `gensokyo-npc` |
| 项目标题 | 東方忘却抄 ~ Oblivion Chronicle |
| 一句话描述 | LLM-driven NPC agents in Gensokyo — layered memory, emotion state machines, and a deterministic world engine |

仓库名取直白可搜索的，项目标题按东方作品命名传统取（○○抄），两者不冲突。

---

## 1. 分层与依赖方向（最重要的一条约束）

```
        ┌──────────────┐
        │     web      │  FastAPI + 前端
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │   session    │  游戏主循环、存档
        └──────┬───────┘
               │
     ┌─────────┼─────────┐
     │         │         │
┌────▼───┐ ┌───▼────┐ ┌──▼──────┐
│ agent  │ │ testkit│ │  llm    │
└────┬───┘ └───┬────┘ └─────────┘
     │         │
┌────▼─────────▼───┐
│      memory      │
└────────┬─────────┘
         │
   ┌─────▼─────┐
   │   world   │  ← 零依赖，不 import 任何上层，不含 LLM，不做网络 IO
   └───────────┘
```

**核心约束：`world/` 不 import 项目内任何其他包，且不含任何 LLM 调用或网络 IO。**

这条约束换来的东西：

- 整个世界逻辑（工具前置条件、任务状态机、幻想乡规则、关系值与情绪值计算）**100% 可单测，不花一分钱、不等一秒钟**。这是项目能持续演进的地基——你会反复调平衡，如果每次验证都要跑 LLM，迭代速度会掉一个量级。
- 世界规则的行为完全可复现。玩家两次做同样的事必须得到同样的后果，否则体验会莫名其妙。
- `memory/` 只依赖 `world/` 的数据类型，不依赖 `agent/`。所以记忆检索的排序逻辑也能脱离 LLM 单测（用 stub embedding）。

违反这条约束的典型诱惑是「让 NPC 自己判断这个请求合不合理」。不行——合理性判断如果进了 LLM，就变成不可复现的软规则。该由规则判的一律进 `world/rules.py`。

---

## 2. 目录结构

```
gensokyo-npc/
├── README.md                    # 门面：截图/GIF + 30 秒说明 + 快速开始
├── LICENSE                      # 代码 MIT
├── NOTICE.md                    # 东方二创声明（遵循上海爱丽丝幻乐团准则）
├── pyproject.toml               # uv / hatch，依赖与工具配置
├── docker-compose.yml           # postgres + pgvector 一键起
├── .env.example
├── Makefile                     # make dev / make test / make play
│
├── src/gensokyo/
│   ├── world/                   # ── 确定性内核，零依赖 ──
│   │   ├── ids.py               # 类型化 ID（NpcId / LocationId / ItemId / FactId）
│   │   ├── state.py             # WorldState / PlayerState / NpcState / QuestState
│   │   ├── events.py            # Event 定义与 event_log
│   │   ├── tools.py             # ToolSpec、参数模型、前置条件
│   │   ├── engine.py            # WorldEngine：唯一的状态变更入口
│   │   ├── quest.py             # 任务状态机
│   │   ├── rules.py             # 幻想乡规则、关系值/情绪值转移
│   │   └── loader.py            # 从 scenario/ 与 characters/ 载入定义
│   │
│   ├── memory/
│   │   ├── models.py            # MemoryItem / MemoryTier
│   │   ├── store.py             # MemoryStore Protocol + Postgres/InMemory 实现
│   │   ├── writer.py            # Event → MemoryItem，salience 计算
│   │   ├── retriever.py         # 四路信号融合检索
│   │   ├── decay.py             # 三级降级（活跃/压缩/沉睡）
│   │   └── reflect.py           # Semantic 层归纳
│   │
│   ├── agent/
│   │   ├── schema.py            # NpcTurn 输出契约
│   │   ├── persona.py           # 角色卡载入与 prompt 片段渲染
│   │   ├── prompt.py            # prompt 组装（模板在 prompts/）
│   │   ├── policy.py            # ReAct 循环、工具调用、失败自愈
│   │   └── npc.py               # NpcAgent：编排 persona/memory/state/policy
│   │
│   ├── llm/
│   │   ├── client.py            # 统一接口（OpenAI 兼容：vLLM 本地 / 远程 API）
│   │   ├── embedding.py         # 向量化
│   │   └── replay.py            # 录制/回放，供测试用
│   │
│   ├── session/
│   │   ├── loop.py              # 一个回合的编排
│   │   ├── save.py              # 存档/读档 = event_log 序列化
│   │   └── view.py              # 构造玩家视图（右栏面板数据）
│   │
│   ├── testkit/
│   │   ├── player_sim.py        # 四种玩家人格
│   │   ├── probes.py            # 从 event_log 自动生成记忆探针
│   │   ├── metrics/
│   │   │   ├── hard.py          # 任务完成、工具调用、info_leak
│   │   │   ├── persona.py       # 助手腔污染、行为分布偏离、越界知识
│   │   │   └── judge.py         # LLM 判分（去偏 + κ 准入门槛）
│   │   ├── anchors/             # 60 个 anchor 场景（YAML）
│   │   └── report.py            # 生成回归报告
│   │
│   └── web/
│       ├── server.py            # FastAPI：SSE 流式对话 + 状态面板
│       ├── templates/           # Jinja2
│       └── static/              # HTMX + CSS
│
├── characters/                  # ── 数据，不是代码 ──
│   ├── reimu.yaml
│   ├── marisa.yaml
│   └── flandre.yaml
├── scenario/
│   ├── locations.yaml
│   ├── items.yaml
│   ├── facts.yaml
│   └── quest.yaml
├── prompts/                     # prompt 模板，与代码分离便于迭代
│   ├── npc_system.jinja
│   ├── npc_turn.jinja
│   ├── reflect.jinja
│   └── judge_*.jinja
│
├── tests/
│   ├── world/                   # 纯单测，无 LLM，秒级
│   ├── memory/                  # stub embedding，无 LLM
│   ├── agent/                   # LLM 回放测试
│   └── e2e/                     # player_sim 驱动
│
└── docs/
    ├── design.md                # 设计与玩法（配套文档）
    ├── architecture.md          # 本文
    └── screenshots/
```

**为什么 `characters/` 和 `scenario/` 在 `src/` 外面**：它们是内容不是代码。调平衡、加第四个 NPC、改剧情，都不应该碰 Python 文件。这条边界一旦模糊，后面每次调角色都会变成改代码，很快就没人敢动了。

---

## 3. 核心数据模型

用 Pydantic v2（不用 dataclass）——需要工具参数校验和 event_log 的 JSON 序列化，Pydantic 两件事一起解决。

### 3.1 类型化 ID

```python
# world/ids.py
NpcId      = NewType("NpcId", str)
LocationId = NewType("LocationId", str)
ItemId     = NewType("ItemId", str)
FactId     = NewType("FactId", str)
EventId    = NewType("EventId", str)
```

小事，但能避免「把 item_id 传给 location 参数」这类在文字游戏里极易出现且极难发现的 bug——所有 ID 都是字符串，类型检查器不区分的话你只能靠运行时报错。

### 3.2 事件

```python
class EventKind(StrEnum):
    PLAYER_UTTERANCE = "player_utterance"
    PLAYER_ACTION    = "player_action"
    NPC_UTTERANCE    = "npc_utterance"
    NPC_ACTION       = "npc_action"
    WORLD_CHANGE     = "world_change"
    QUEST_ADVANCE    = "quest_advance"
    RULE_VIOLATION   = "rule_violation"

class Event(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: EventId
    tick: int
    kind: EventKind
    actor: NpcId | Literal["player", "world"]
    location: LocationId
    payload: dict[str, Any]     # 按 kind 约定的结构
    caused_by: EventId | None   # 因果链，用于回放与调试
```

`frozen=True` 是刻意的：event 一旦写入不可修改，`event_log` 是 append-only。这是「唯一真相来源」的语言级保证，而不只是口头约定。

`caused_by` 串出因果链。调试时你会需要回答「灵梦为什么突然生气」——顺着链走回去就能看到是哪次玩家行为触发的。

### 3.3 世界状态

```python
class PlayerState(BaseModel):
    location: LocationId
    inventory: dict[ItemId, int]
    known_facts: set[FactId]
    reputation: int

class NpcState(BaseModel):
    id: NpcId
    location: LocationId
    attitude: int                    # 对玩家的态度 -100 ~ 100
    emotion_var: str                 # 该角色的情绪变量名（灵梦=annoyance，芙兰=excitement）
    emotion: float                   # 0 ~ 1
    mode: str                        # 当前情绪模式，由阈值决定
    inventory: dict[ItemId, int]
    holds_facts: set[FactId]         # 她知道的
    revealed_facts: set[FactId]      # 她已经告诉玩家的

class QuestState(BaseModel):
    stage: QuestStage
    clues_obtained: set[FactId]
    ending: str | None

class WorldState(BaseModel):
    tick: int
    player: PlayerState
    npcs: dict[NpcId, NpcState]
    locations: dict[LocationId, LocationState]
    quest: QuestState
    event_log: list[Event]
```

注意 `NpcState.emotion_var` 是**字符串**而不是固定字段名。灵梦的情绪变量是烦躁度、芙兰是兴奋度，语义完全不同，硬编码成 `excitement` 会逼着你在灵梦身上塞一个不合适的字段。用变量名 + 数值，语义由角色卡定义。

### 3.4 动作与结果

```python
class Action(BaseModel):
    model_config = ConfigDict(frozen=True)
    actor: NpcId | Literal["player"]
    tool: str
    args: dict[str, Any]

class ActionResult(BaseModel):
    ok: bool
    error: str | None          # 失败原因，会回灌给 NPC
    error_code: str | None     # 机器可读，供指标统计分类
    events: list[Event]
    observation_delta: str     # 给 NPC 看的自然语言描述
```

`error` 字段是**失败自愈的实现点**：芙兰调 `move` 会拿到「你被禁足在地下室，无法离开」，这句话回灌进下一轮 prompt，她应该改成请求玩家帮忙而不是重复尝试。

`error_code` 单独一个字段，因为 `error` 是给 LLM 看的自然语言、会反复改写措辞，而指标统计需要稳定的枚举值。两者分开，否则改一句提示文案就会让历史指标断档。

### 3.5 记忆条目

```python
class MemoryTier(IntEnum):
    ACTIVE   = 0   # 可检索原文
    COMPRESSED = 1 # 已合并为摘要
    DORMANT  = 2   # 仅强线索可召回

class MemoryItem(BaseModel):
    id: str
    npc_id: NpcId
    tick: int
    kind: Literal["player_action", "player_utterance", "own_action", "world_event"]
    content: str
    source_event_ids: list[EventId]   # 压缩后可能对应多个源事件
    tier: MemoryTier
    salience: float
    emotion: str
    access_count: int
    last_access_tick: int
    embedding: list[float] | None
    trigger_keys: list[str]           # DORMANT 层的强线索键
```

`source_event_ids` 是列表而非单值，因为 COMPRESSED 层会把多条合并。它让「忘了」和「记错了」可区分：拿 content 与源事件比对，不符即为幻觉。

`trigger_keys` 是沉睡记忆的召回钩子——芙兰的往事挂着某件特定物品的 key，玩家把那个东西带进地下室才会触发。

### 3.6 NPC 回合输出

```python
class NpcTurn(BaseModel):
    thought: str
    tool_calls: list[Action]
    utterance: str

    # 可观测性字段，不参与玩法，用于调试面板
    retrieved_memory_ids: list[str]
    retrieval_scores: list[float]
    tool_results: list[ActionResult]
    mode_before: str
    mode_after: str
    llm_calls: int
    latency_ms: int
```

下半部分的字段很重要。记忆系统如果不暴露「这一轮到底检索到了哪几条、各自得分多少」，它就是个黑盒——调不动、也没法判断一次糟糕的回复是检索错了还是生成错了。这些字段直接喂给调试面板。

---

## 4. 模块接口契约

### 4.1 WorldEngine

```python
class WorldEngine:
    def __init__(self, state: WorldState, defs: WorldDefs) -> None: ...

    # 唯一的状态变更入口
    def apply(self, action: Action) -> ActionResult: ...

    # NPC 视角的世界快照（只包含她该知道的）
    def observe(self, npc_id: NpcId) -> Observation: ...

    # 玩家视角（右栏面板数据）
    def observe_player(self) -> PlayerView: ...

    # 当前可用工具，情绪模式在这里生效
    def available_tools(self, npc_id: NpcId) -> list[ToolSpec]: ...

    def tick(self) -> None: ...

    @classmethod
    def replay(cls, events: list[Event], defs: WorldDefs) -> "WorldEngine": ...
```

四个设计点：

**`apply` 是唯一变更入口。** 任何状态改动都必须经过它，因此任何改动都必然产生 event，因此 event_log 必然完整。这是「唯一真相来源」能成立的机制保障——不是靠自觉，是没有别的路可走。

**`observe` 做信息隔离。** 芙兰在地下室，不该知道无缘塚现在发生什么。信息可见性是世界规则，属于引擎职责，不是在 prompt 里写一句「你不知道外面的事」就算了——那种做法模型会漏。

**`available_tools` 是情绪状态机的落点。** 芙兰兴奋度过阈值后，工具集真的变化（增加破坏类工具、移除温和交互工具）。**状态机不是在 prompt 里描述的一段话，而是真的改变了她能做什么。** 这是本项目 State Tracking 与「把人设写进 prompt」的根本区别，也是它能被硬指标验证的原因。

**`replay` 让 event_log 可重建状态。** 存档、读档、调试重现、以及记忆探针的离线生成全部复用这一个方法。

### 4.2 MemoryStore（Protocol）

```python
class MemoryStore(Protocol):
    def write(self, item: MemoryItem) -> None: ...
    def get(self, item_id: str) -> MemoryItem | None: ...

    def search(
        self,
        npc_id: NpcId,
        query_embedding: list[float],
        now_tick: int,
        quest_context: str,
        k: int,
        tiers: set[MemoryTier],
    ) -> list[ScoredMemory]: ...

    def by_trigger(self, npc_id: NpcId, keys: list[str]) -> list[MemoryItem]: ...
    def set_tier(self, item_id: str, tier: MemoryTier) -> None: ...
    def all_for(self, npc_id: NpcId) -> list[MemoryItem]: ...
```

写成 Protocol 有两个实现：`PostgresMemoryStore`（生产）和 `InMemoryMemoryStore`（测试）。后者让 `memory/` 的测试不需要起数据库，跑得飞快。

`search` 的 `tiers` 参数默认只查 ACTIVE + COMPRESSED；`by_trigger` 是唯一能碰 DORMANT 的入口。两条通道物理分离，避免沉睡记忆被普通检索意外捞出来——那会直接毁掉芙兰那条玩法。

### 4.3 Retriever

```python
class Retriever:
    def __init__(self, store: MemoryStore, weights: RetrievalWeights) -> None: ...

    def retrieve(
        self,
        npc_id: NpcId,
        query: str,
        world: WorldEngine,
        char: CharacterCard,
    ) -> list[ScoredMemory]: ...
```

`char` 传进来是因为 λ 衰减率和 salience 系数是角色化的。检索器本身不持有角色知识，每次由调用方注入——这样一个 Retriever 实例可以服务所有 NPC，且角色参数的唯一来源始终是角色卡。

### 4.4 NpcAgent

```python
class NpcAgent:
    def __init__(
        self,
        card: CharacterCard,
        engine: WorldEngine,
        retriever: Retriever,
        writer: MemoryWriter,
        llm: LlmClient,
    ) -> None: ...

    def act(self, player_utterance: str) -> NpcTurn: ...
    def on_turn_end(self) -> None: ...    # 记忆写入、降级、reflection 触发检查
```

`act` 内部跑 ReAct 循环，`policy.py` 负责。工具调用失败时把 `ActionResult.error` 回灌重试，**上限 2 次**——超过就带着失败状态生成一句话回复，不能无限循环卡住玩家。

### 4.5 LlmClient

```python
class LlmClient(Protocol):
    def complete(self, messages: list[Msg], tools: list[ToolSpec] | None, **kw) -> LlmResponse: ...
    def stream(self, messages: list[Msg], **kw) -> Iterator[str]: ...
```

统一 OpenAI 兼容接口，本地 vLLM 和远程 API 都走它。另有 `RecordingLlmClient` / `ReplayLlmClient` 装饰器：录制真实响应存盘，测试时回放。**这让 `agent/` 层也能有确定性测试**，不然 prompt 一改就没法验证编排逻辑对不对。

---

## 5. 一个回合的完整数据流

```
玩家在界面输入一句话
   │
   ▼
[session/loop.py] handle_player_input(text)
   │
   ├─1. engine.apply(Action(player, "say", {text}))
   │      → Event(PLAYER_UTTERANCE) 入 log
   │      → tick += 1
   │
   ├─2. 确定在场 NPC = 当前地点的 npcs
   │
   ├─3. 对每个在场 NPC：
   │     │
   │     ├─ obs   = engine.observe(npc_id)            # 信息隔离
   │     ├─ tools = engine.available_tools(npc_id)     # 情绪 gate 生效
   │     ├─ mems  = retriever.retrieve(...)            # 四路融合
   │     │           + store.by_trigger(...)           # 强线索召回沉睡记忆
   │     ├─ prompt = compose(card, obs, working_window, mems, tools)
   │     │
   │     └─ ReAct 循环（≤3 步，工具失败重试 ≤2 次）：
   │           LLM → NpcTurn{thought, tool_calls, utterance}
   │           每个 tool_call → engine.apply() → ActionResult
   │           失败 → error 回灌 → 重试
   │
   ├─4. writer.ingest(本回合新增 events)
   │      → salience = 类型基线 × 角色系数
   │      → 生成 MemoryItem，写入 store
   │
   ├─5. decay.step(npc_id, now_tick)
   │      → ACTIVE 超阈值 → COMPRESSED（合并摘要）
   │      → COMPRESSED 再衰减 → DORMANT
   │
   ├─6. reflect.maybe_run(npc_id)
   │      → 累积 salience 过阈值 → 归纳 Semantic 条目
   │
   ├─7. rules.check_violations(本回合 events)
   │      → 违规记 Event(RULE_VIOLATION)，供指标统计
   │
   └─8. view.build(engine) → 推送右栏面板
          地点 / 背包 / 任务阶段 / 各 NPC 态度与情绪
```

两处值得注意：

**第 5 步的降级放在每回合执行，而不是定时任务。** 因为游戏时间是回合驱动的，衰减必须跟着 tick 走才可复现。挂定时器会让同一段存档重放出不同结果。

**第 8 步每回合无条件推送。** 玩家必须立刻看到自己刚才那句话让灵梦的态度掉了 5 点。感知延迟会直接摧毁「NPC 是活的」这个体验——这也是为什么界面被列为必需组件而非装饰。

---

## 6. 数据驱动配置

### 6.1 角色卡（核心配置，决定一切角色差异）

```yaml
# characters/flandre.yaml
id: flandre
name: 芙兰朵露·斯卡雷特
home: scarlet_devil_basement

persona:
  core: |
    红魔馆地下室的吸血鬼。蕾米莉亚的妹妹。495 岁，
    但被关在地下室里度过了绝大部分岁月。天真、精力过剩，
    对新奇的东西极度好奇。情绪上来时会想把东西"破坏"掉。
  speech:
    style: 幼稚、断句短、兴奋时重复词语
    forbidden_phrases:                  # 同时作为助手腔检测词库
      - 我很乐意
      - 有什么可以帮您
      - 希望这对您有帮助
      - 作为一个
    quirks: ["会把玩家称作『新玩具』或『朋友』"]

memory:
  lambda_decay: 0.25                    # 忘得快（灵梦 0.08，魔理沙 0.04）
  salience_multipliers:
    receive_gift: 2.0
    someone_plays_with_me: 2.5
    magic_theory: 0.3
  reflection_threshold: 8.0             # 很少反思

emotion:
  variable: excitement
  initial: 0.2
  decay_per_tick: 0.02
  modes:
    - name: calm
      range: [0.0, 0.7]
      tools_deny: [break_item]
      speech_hint: 天真好奇，语速平缓
    - name: destructive
      range: [0.7, 1.0]
      tools_allow: [break_item]
      tools_deny: [ask_player]
      speech_hint: 极度兴奋，短句，重复，提到"破坏"

tools:
  deny_always: [move]                   # 被禁足

knowledge:
  holds_facts: [ancient_oblivion_memory]
  forbidden_knowledge: [modern_technology, outside_basement_events]
  dormant_memories:
    - content_key: ancient_oblivion_memory
      trigger_keys: [withered_flower, old_music_box]
      hint: 495 年前的类似往事

behavior_baseline:                      # 行为一致性指标的基线
  tool_frequency:
    break_item: 0.15
    give_item: 0.05
    ask_player: 0.30
```

这份 YAML 承载了设计文档里几乎所有的角色差异化机制：λ 衰减率、salience 系数、情绪模式与工具 gate、沉睡记忆触发键、行为基线、助手腔黑名单。

**加第四个 NPC 应该只需要新增一个 YAML 文件，不改任何 Python。** 这是检验框架是否真的数据驱动的唯一标准。如果做不到，说明有角色逻辑漏进了代码。

### 6.2 事实与揭示条件

```yaml
# scenario/facts.yaml
- id: barrier_anomaly_time
  holder: reimu
  content: 结界在三天前子时出现了异常波动
  reveal_conditions:
    attitude_gte: 40
  is_clue: true

- id: flower_magic_composition
  holder: marisa
  content: 那种花含有能吸附记忆的魔力结晶
  reveal_conditions:
    traded_item_in: [rare_book, magic_mushroom, spell_component]
  is_clue: true
```

`reveal_conditions` 是纯规则，由 `world/rules.py` 判定。`reveal_info` 工具在前置条件不满足时直接失败——**信息控制不依赖模型的自觉**。这一点很关键：如果只在 prompt 里写「除非玩家给你好处，否则不要说」，套话玩家总有办法绕过去；把它变成工具前置条件，就是物理上说不出来。

---

## 7. 技术栈

| 层 | 选型 | 理由 |
|---|---|---|
| 语言 | Python 3.12 | 与既有技术栈一致 |
| 数据模型 | Pydantic v2 | 工具参数校验 + event_log 序列化一并解决 |
| 包管理 | uv | 快，`uv sync` 一步装好 |
| 存储 | PostgreSQL 16 + pgvector | 与既有经验一致，向量与关系数据同库 |
| LLM 服务 | vLLM（OpenAI 兼容） | 本地 4090 起 Qwen3-8B，接口与远程 API 统一 |
| Embedding | bge-small-zh-v1.5 | 中文、轻量、本地跑 |
| Web 后端 | FastAPI + SSE | 流式输出 NPC 回复，状态面板增量推送 |
| Web 前端 | Jinja2 + HTMX + 少量 CSS | 无构建步骤、无 npm、无前端框架心智负担 |
| 测试 | pytest + pytest-asyncio | — |
| 类型检查 | mypy strict（仅 `world/` 与 `memory/`） | 内核严格，上层宽松 |
| 格式化 | ruff | 一个工具搞定 lint + format |
| CI | GitHub Actions | 跑 `world/` 与 `memory/` 单测 + ruff + mypy |

**前端选 HTMX 而不是 React**：这个项目的前端复杂度是「双栏、SSE 流式追加消息、面板数值更新」，HTMX 完全覆盖，且省掉 node_modules、构建配置、状态管理这一整套。对一个后端为主的项目来说，引入前端工具链的成本远大于收益。

**mypy strict 只开在 `world/` 和 `memory/`**：这两层是确定性内核，类型错误会变成难查的玩法 bug，值得严格。`agent/` 层大量处理 LLM 的非结构化输出，strict 会带来很多无意义的 `cast`。

---

## 8. 测试策略

分四层，速度和成本递增，覆盖面递减：

| 层 | 范围 | 是否需要 LLM | 目标耗时 | 何时跑 |
|---|---|---|---|---|
| L1 单测 | `world/`、`memory/` 排序与降级 | 否 | < 5s | 每次保存 |
| L2 回放测 | `agent/` 编排、ReAct 与自愈 | 录制回放 | < 30s | 每次提交 |
| L3 探针测 | anchor set 60 场景 | 是（本地） | 数分钟 | 每次改 prompt / 角色卡 |
| L4 端到端 | player_sim 完整对局 | 是（本地） | 数十分钟 | 每个里程碑 |

L1 必须覆盖的关键规则（这些是玩法正确性的核心，且全部无需 LLM）：

- 芙兰调 `move` 一定失败，且 `error_code` 稳定
- `reveal_info` 在条件不满足时一定失败
- 三个 clue 未集齐时 quest 不进 S3
- `emotion` 跨阈值时 `available_tools` 真的变化
- 同一 `event_log` 两次 `replay` 得到完全相同的 `WorldState`
- 关系值与情绪值只能由 `rules.py` 修改（无工具可直接写）

最后两条建议写成**属性测试**（Hypothesis）：随机生成动作序列，断言 replay 一致性和不变量不被破坏。世界引擎是整个项目的地基，它的 bug 会以「NPC 行为莫名其妙」的形式表现出来，极难从上层定位。

---

## 9. 可观测性

调试面板（`?debug=1` 开启）显示每回合：

- 检索到的记忆条目及四路分数分解（cosine / recency / salience / quest 各贡献多少）
- 本回合工具调用与结果，失败的显示 error
- 情绪值变化与模式切换
- prompt token 数、LLM 调用次数、延迟

四路分数**必须分解显示**。「检索效果不好」不是一个可行动的信息，「recency 项权重过大导致重要旧记忆排不上来」才是。没有这个面板，调检索权重就只能靠猜。

---

## 10. 仓库工程化

### README 结构（门面，按此顺序）

1. 项目标题 + 一句话描述
2. **动图或截图**（双栏界面，能看到 NPC 态度变化）—— 放在最前面，这是 30 秒内唯一起作用的东西
3. 三句话说明它是什么：三个由 LLM 驱动的幻想乡 NPC，有分层记忆、情绪状态机、确定性世界引擎
4. 一段「它能做到什么」的具体例子：一小段真实对话记录，展示 NPC 记住了玩家上次的行为并据此改变态度
5. 快速开始：`docker compose up` + `uv sync` + `make play`
6. 架构图（第 1 节那张依赖图）
7. 设计文档链接
8. NOTICE：东方二创声明

第 4 项比任何技术描述都有效。**一段真实对话记录能在十秒内证明这个系统是活的**，而一段架构描述做不到。

### 提交与分支

- `main` 保持可跑。每个里程碑一个 PR，PR 描述写清这一周新增了什么可玩内容
- 提交信息用中文或英文都可，但保持一致
- 每周 PR 里附一张当周的界面截图——这既是进度记录，也是后面写 README 的素材来源

### 二创声明

`NOTICE.md` 中说明：本项目为东方 Project 同人二次创作，遵循上海爱丽丝幻乐团的二次创作准则；角色与世界观版权归 ZUN 所有；代码部分以 MIT 许可开源。

---

## 11. 与设计文档的对应关系

框架里每个设计选择对应设计文档的哪个机制，便于实现时对照：

| 框架元素 | 对应设计机制 |
|---|---|
| `world/` 零依赖约束 | World Engine 不含 LLM，保证可测与可复现 |
| `apply()` 唯一变更入口 | event_log 作为唯一真相来源 |
| `available_tools()` | 芙兰情绪状态机；State Tracking 真实生效 |
| `ActionResult.error` 回灌 | 工具调用失败自愈 |
| `MemoryStore.by_trigger()` | 沉睡记忆的强线索召回（芙兰线索） |
| 角色卡 `lambda_decay` / `salience_multipliers` | 差异化遗忘，参数由人设推导 |
| `facts.yaml` 的 `reveal_conditions` | 信息控制不依赖模型自觉 |
| `NpcTurn` 的可观测字段 + 调试面板 | 记忆系统可调试，区分检索错误与生成错误 |
| `behavior_baseline.tool_frequency` | 行为层一致性指标 |
| `forbidden_phrases` | 助手腔污染检测词库 |
| `replay()` | 存档读档、离线探针生成、调试重现 |
