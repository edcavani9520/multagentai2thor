# 基于中转智能体的多机器人闭环任务重规划机制

## 摘要

本文档描述 `--relay-mode --closed-loop-replan` 模式下的多机器人任务执行机制。该机制面向由 primary agent 接收用户任务、但任务可能需要其他机器人协作完成的场景。系统首先将自然语言任务解析为结构化任务意图，然后在每个意图步骤上进行闭环决策：若 primary agent 满足动作前置条件，则直接由 primary agent 执行；若 primary agent 无法完成，则调用中转智能体根据当前可用的全局摘要、已观察机器人状态、候选执行者证据和硬验证结果选择 executor robot。executor 选定后，系统再从 executor 的视觉视角生成该步骤的语义计划，并将其 grounding 为 simulator 可执行动作。

该设计的核心特点是逐步闭环重规划，而不是一次性生成完整任务计划。每个步骤都经历“选择执行者、生成执行计划、执行动作、合并反馈”的循环，从而允许系统在物体持有状态、可见性和执行反馈变化时更新后续决策。

## 系统组成

系统由五个主要模块构成。

**任务意图解析模块**将用户原始任务 `task` 转换为结构化 `task_intent`。`task_intent` 包含请求动作、请求物体以及可选的 `intentSteps`。在当前实现中，系统会调用 Qwen 的 tool-call 接口生成工具调用记录，但最终结构化意图由本地 `extract_task_intent` 工具执行得到。

**闭环控制器**负责遍历 `task_intent.intentSteps`。对每个步骤，控制器构造单步自然语言描述 `step_task`、单步意图 `step_intent` 和路由提示 `step_plan_hint`。其中 `step_plan_hint` 仅用于执行者选择和硬验证，并不等价于视觉模型生成的真实语义计划。

**Primary fast path 验证器**负责判断 primary agent 是否已经能够完成当前步骤。该判断由硬验证函数完成，而非由大模型主观判断。验证条件包括目标物体是否可见、动作 affordance 是否满足、机器人是否持有正确物体、目标 receptacle 是否可见，以及当前状态是否允许动作执行。

**中转智能体**在 primary agent 无法完成当前步骤时被调用。中转智能体由大模型驱动，但只通过受控工具获取证据和提交决策。它可以查询全局摘要、评估候选执行者、观察指定机器人、选择 executor 或报告失败。程序对 executor 选择和 failure 报告均执行硬验证，从而防止大模型选择不存在、未观察或不满足执行条件的机器人。

**Executor 视角语义规划与执行模块**在 executor 选定后运行。该模块使用 executor 当前图像和可见物体列表生成真实的 `step_semantic_plan`，并将该计划 grounding 为带有 `objectId` 的动作 payload。payload 发送至 `execute_actions` 接口执行，执行反馈随后被合并回机器人 observation、held object 和 inventory 状态中。

## 核心数据结构

`task` 是用户输入的原始自然语言任务，例如“put the mug in the cabinet”。它是整个流程的根输入。

`task_intent` 是从 `task` 中抽取得到的结构化任务意图。它通常包含 `requestedAction`、`requestedObjectType` 和 `intentSteps`。例如，一个放置任务可能被表示为先打开 receptacle，再执行 `PutObject`。

`step` 是 `task_intent.intentSteps` 中的单步动作。闭环控制器以 step 为单位进行执行者选择、规划和执行。

`step_task` 是由 `step` 反写得到的单步自然语言描述，用于后续视觉规划 prompt 和日志输出。该字段使每个步骤能够被独立交给 executor 视角的语义规划器。

`step_intent` 是由单个 `step` 包装得到的小型任务意图，其结构与 `task_intent` 保持一致。它用于复用已有的验证器、中转智能体输入和规划约束接口。

`step_plan_hint` 是由 `step` 构造的轻量路由提示。它具有类似 semantic plan 的 JSON 形状，但不由视觉模型生成，也不直接用于执行。其作用是让 fast path 验证器和中转智能体能够复用以 semantic plan 为输入的硬验证逻辑。

`step_semantic_plan` 是 executor 选定后，由 Qwen 基于 executor 图像和 executor 可见物体生成的真实语义计划。该计划随后被验证并 grounding 成 `execute_actions` 可执行的动作序列。

## 闭环重规划流程

系统首先对 primary agent 执行初始 probe，以获得 primary 视角下的图像、可见物体、机器人状态和场景元数据。随后，任务意图解析模块根据用户任务和当前可用物体类别生成 `task_intent`。

进入闭环控制器后，系统从 `task_intent` 中取出 `intentSteps`。如果没有显式步骤，则根据 `requestedAction` 和 `requestedObjectType` 构造单步任务。对于 `PutObject` 类型任务，系统会在 relay 模式下额外查询可能的物体持有状态，并根据已知 held object 信息补全隐含的 pickup/open 前置步骤。

对于每个 step，控制器首先检查该步骤是否已经满足。例如，如果某个机器人已经持有待 pickup 的物体，则对应 `PickupObject` 步骤可以被跳过；如果目标 cabinet 已经打开，则 `OpenObject` 步骤也可以被跳过。跳过步骤会被记录到 `closed_loop_trace` 中。

若步骤尚未满足，系统执行 primary fast path 验证。若 primary agent 通过硬验证，则当前 executor 直接设为 primary agent，系统不会调用中转智能体。若 primary agent 验证失败，系统先确定性地收集所有已知机器人的当前 observation，再统一构造全局摘要和候选证据表，最后调用中转智能体进行跨机器人协调。

executor 选定后，系统使用 executor 的图像和物体列表生成 `step_semantic_plan`。该计划必须与 `step_intent` 保持一致，并满足动作 affordance 与状态前置条件。验证通过后，系统将 object type grounding 为 simulator 中的 objectId，并构造 `execute_actions` payload。payload 顶层包含 `stop_on_failure: false`，以保证动作序列失败时 simulator 不因默认停止策略造成不一致。

执行成功后，系统合并执行反馈，更新 executor 的 held object、inventory、机器人状态和可见性摘要，然后进入下一 step。若执行失败，则系统返回 `needs_upstream_planning`，并在输出 JSON 中记录失败步骤、失败原因和已收集的 trace。

## 中转智能体机制

中转智能体的入口是 `route_with_relay_agent`，其内部调用 `run_relay_agent`。主线采用“先收集、后判断”的两阶段协议：程序先遍历尚无当前快照的 known robots，通过 receiver 获取第一视角图像与 metadata，提取位置、可见物体、affordance、持有物和 inventory，并完成统一候选硬验证；随后中转智能体综合所有机器人视图和结构化证据选择最佳 executor。

底层 agent runtime 支持五类工具，以便独立测试和按需模式复用。

`inspect_global_scene()` 返回当前任务、路由步骤、已知机器人、已观察机器人、visibility unknown 机器人、物体可见性摘要以及各机器人 observation summary。该工具用于让中转智能体获得结构化全局概况。

`evaluate_executor_candidates()` 返回候选执行者证据表。该表包含每个已观察机器人的可执行性、硬验证原因、目标物体可见性、目标 receptacle 可见性、持有状态、inventory、位置和目标距离。该工具只提供证据，不替中转智能体做最终选择。

`observe_robot(robot_id)` 用于按需模式补充指定机器人的 observation。主线 relay 在进入中转决策前已经完成全量采集，因此不会把该工具暴露给中转模型。

`select_executor(robot_id, reason)` 是中转智能体提交 executor 选择的终止工具。程序会再次调用硬验证函数确认该 robot 是 known、observed 且满足当前 step 的执行条件。只有硬验证通过后，选择才被接受。

`report_failure(failure_code, reason)` 是中转智能体报告无法协调完成任务的终止工具。程序会验证 failure 是否合理，例如在仍存在关键 visibility unknown 机器人时不能轻易报告目标不可见。

主线全量采集模式只向中转模型暴露 `select_executor` 和 `report_failure`。`inspect_global_scene`、`evaluate_executor_candidates` 的结果已经预先写入首次输入，`observe_robot` 也已由程序完成，因此模型不会再通过多轮工具调用逐个轮询机器人。

## 候选执行者评估

候选执行者评估由 `evaluate_relay_executor_candidates` 实现。其输入包括当前 task、`step_plan_hint`、`step_intent`、`object_visibility_map`、`agent_observations`、`known_robot_ids` 和 primary robot id。

评估器首先确定当前 step 的动作类型、请求物体和目标 receptacle。对于 `PickupObject`，距离目标通常是请求物体；对于 `PutObject`，距离目标通常是目标 receptacle。随后，评估器遍历所有已经存在 observation 的机器人，并对每个机器人调用硬验证函数判断其是否可执行当前 step。

输出中的 `candidate_scores` 包含每个候选机器人的证据，例如 `executable`、`validation`、`can_see_requested_object`、`can_see_target_receptacle`、`holds_requested_object`、`held_object_type`、`inventory_object_types` 和 `distance_to_target`。输出中的 `candidate_executor_robot_ids` 表示当前证据下通过硬验证的机器人集合。

当前实现中，候选评估器评估全量采集成功的机器人。采集失败的机器人会保留在 `observation_errors`，并参与最终失败类型验证；系统不会把采集失败误判为“目标不可见”。该设计既避免将全局物体存在性误认为某个机器人当前可见性，也避免模型在证据不足时反复轮询。

## 输出与可观测性

系统最终输出 JSON，其中包含任务 id、primary robot、known robots、queried robots、`task_intent`、`intent_steps`、`closed_loop_trace`、每步 payload 和执行结果摘要。对于经过 relay 的步骤，trace 中会记录 `relay_explanation`，包括 primary 无法执行的原因、中转协调结果以及候选摘要。

控制台的 `stderr` 会打印简短解释，例如 primary 为什么不能执行当前 step、中转智能体为什么能够或不能协调完成任务，以及候选机器人摘要。结构化 JSON 仍输出到 `stdout`，便于脚本调用和后续分析。

## 失败处理

系统可能在多个阶段返回 `needs_upstream_planning`。当意图步骤数超过 `--max-replan-steps` 时，系统会直接报告失败。当中转智能体无法选择合法 executor 或报告 failure 通过验证时，系统返回 relay failure。当 executor 视角的 `step_semantic_plan` 与 `step_intent` 不一致、缺少必要 objectId 或不满足状态前置条件时，系统返回规划验证失败。当 `execute_actions` 返回失败状态时，系统记录失败 step 并停止后续闭环步骤。

这些失败均保留当前已知机器人、已查询机器人、最终物体可见性摘要和闭环 trace，使上游系统或人工调试能够判断失败来自感知、协调、规划、grounding 还是执行接口。

## 当前实现边界与改进方向

当前 relay 机制仍依赖已观察机器人的 observation 来判断可见性和可执行性。虽然初始 probe 可能包含全局 `state.objects` 和 `state.robots`，但全局物体列表并不必然等价于每个机器人当前视角可见的物体集合。因此，系统不会直接假设所有机器人都能看到全局 state 中的目标物体。

每次 primary fast path 失败都会先补齐当前尚未取得的 per-robot observation。若未来 receiver 能在一次请求中返回可靠的所有机器人图像和 per-robot visibility map，可以把当前逐机器人 probe 的采集阶段替换成单次批量快照，但中转智能体的“完整证据后决策”接口无需改变。

另一个边界是 `step_task` 由结构化 step 反写成自然语言，可能带来介词或空间关系的语义损失。例如 `put mug in cabinet` 可能被反写为 `put the mug on the cabinet`。后续可以将空间关系显式加入 `task_intent`，减少从结构化动作回写自然语言时的信息损失。

最后，`step_plan_hint` 的命名容易与视觉模型生成的 `step_semantic_plan` 混淆。更清晰的工程命名应为 `routing_plan_hint`，以强调其只服务于路由和验证，而非最终执行计划。

## 图示

本文档配套时序图见：

- Mermaid 源文件：`docs/figures/relay_closed_loop_sequence.mmd`
- 矢量图文件：`docs/figures/relay_closed_loop_sequence.svg`
