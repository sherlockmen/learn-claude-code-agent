"""
在01节和02节的基础上我们以及有了一个具有可扩展性工具的agent loop，agent已经学会了读文件、写文件、跑命令
但是问题还是会出现以下的几个问题：
1. 多步任务容易走一步忘一步，像小熊掰玉米一样，掰一个扔一个
2. 重复走已经走过的步骤，明明已经扫描过的文档，还会再扫描一次，大量消耗token
3. 在分析完用户问之后，一口气列出很多步骤，但是后续不按照步骤走，开始即兴发挥

出现这些问题的主要原因是：模型虽然“能想”，但是它当前的注意力始终受到上下文的影响
所以如果没有一块显式的、稳定的、可反复更新的计划状态，在大任务情况下模型的注意力就会很容易漂移

所以这一小节我们需要给agent补充上一个： 可不断自行更新的计划列表，让 agent 把当前会话里的计划外显出来，并且持续更新。

请注意：我们的这个计划列表并不是任务系统，计划列表与任务系统的主要区别有以下几点：
1. 任务列表是当前会话的轻量计划，用来帮助模型持续聚焦下一步任务；任务系统是一个多agent共用的工作图
2. 任务列表可以随任务推进不断改写；任务系统是一个持久化的任务模板
3. 任务列表不可以后台运行进行任务管理；任务系统可以再后台运行进行任务管理

总结起来 这一章节我们整个的简单结构如下：
用户提出大任务
   |
   v
模型先写一份当前计划
   |
   v
计划状态
  - [ ] 还没做
  - [>] 正在做
  - [x] 已完成
   |
   v
每做完一步，就更新计划

完整的一个agent_todo_write系统的逻辑图
+--------+      +-------+      +---------+
|  User  | ---> |  LLM  | ---> | Tools   |
| prompt |      |       |      | + todo  |
+--------+      +---+---+      +----+----+
                    ^                |
                    |   tool_result  |
                    +----------------+
                          |
              +-----------+-----------+
              | TodoManager state     |
              | [ ] task A            |
              | [>] task B  <- doing  |
              | [x] task C            |
              +-----------------------+
                          |
              if rounds_since_todo >= 3:
                inject <reminder> into tool_result
"""

#---------------------------关键名词解释-------------------------------
"""
1. 会话内规划：为了完成当前这次请求，先把接下来的步骤写出来，并在执行的过程中不断更新
2. todo：模型用来写入当前计划的一条入口
3. active step： 模型当前正在执行的那一步
4. 提醒：在连续几轮交互的过程中模型都忘记更新计划列表，则需要对模型进行提醒，告知模型需要进行更新计划列表
"""

#---------------------------关键数据结构-------------------------------

# 1. PlanItem 计划列表的最小元素
"""
{
    "content": "Read the failing test",
    "status": "pending" | "in_progress" | "completed",
    "activeForm": "Reading the failing test",
}
content： 当前步骤需要做的事项
status： 当前步骤正在处于什么状态
activeForm： 同一个任务在“进行中”状态下的展示文案，用来把“做某事”变成“正在做某事”，以上述例子为例，原始任务为"读取失败测试"，当到这一步执行时就会显示为"正在读取失败测试"

"""

# 2. PlanningState 当前这次任务的“进度记录本”，用来让 Agent 知道自己计划了什么、正在做什么、做完了什么，以及有没有太久没更新计划。
"""
{
    "items": [
        {"content": "读取文件", "status": "completed"},
        {"content": "分析问题", "status": "in_progress"},
        {"content": "输出结论", "status": "pending"}
    ],
    "rounds_since_update": 0
}
rounds_since_update：连续多少轮过去了，模型还没有更新这份计划。
"""

# 3. in_progress 状态约束
"""
有且仅有一个任务可执行，强制模型聚焦当前一步
"""

#---------------------------最小实现-------------------------------

# STEP 1 定义一个计划管理函数 用于处理和更新计划列表
"""
class TodoManager:
    def __init__(self):
        self.items = []
"""

# STEP 2 给定一个更新计划列表的方法 允许模型整体更新当前计划
"""
def update(self, items: list) -> str:
    validated = []
    in_progress_count = 0

    for item in items:
        status = item.get("status", "pending")
        if status == "in_progress":
            in_progress_count += 1
        validated.append({
            "content": item["content"],
            "status": status,
            "activeForm": item.get("activeForm", ""),
        })

    if in_progress_count > 1:
        raise ValueError("Only one item can be in_progress")

    self.items = validated
    return self.render()
"""

# STEP 3 将计划渲染成可读文本
"""
def render(self) -> str:
    lines = []
    for item in self.items:
        marker = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "completed": "[x]",
        }[item["status"]]
        lines.append(f"{marker} {item['content']}")
    return "\n".join(lines)
"""

# STEP 4 将_todo工具接入工具列表
"""
TOOL_HANDLERS = {
    "read_file": run_read,
    "write_file": run_write,
    "edit_file": run_edit,
    "bash": run_bash,
    "todo": lambda **kw: TODO.update(kw["items"]),
}
"""

# STEP 5 最终如果多次没有更新计划列表则提醒模型对计划列表进行更新
"""
if rounds_since_update >= 3:
    results.insert(0, {
        "type": "text",
        "text": "<reminder>Refresh your plan before continuing.</reminder>",
    })
"""
