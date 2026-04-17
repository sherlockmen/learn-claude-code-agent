"""
循环是agent的最小实现方式，本质上一个agent就是一个循环组成的。

语言模型本身只会生成下一段内容
    它不会自己执行命令操作，例如： 打开文件、运行命令、观察报错、把工具结果再接着用于下一步的推理

这个时候就需要一层代码在其中反复的做以下事情：
    发送请求给模型
        -> 发现模型想调用的工具
        -> 真的去执行工具
        -> 把结果再喂回给模型
        -> 继续下一轮

所以一个agent loop的核心就是把模型 + 工具连成一个能持续推荐任务的主循环

名词解释：
    loop 循环 只要任务还没做完，系统就继续重复同一套步骤
    turn 一轮 循环的轮数 一轮就是执行一遍循环
    tool_result 工具执行结果 这个结果要重新写回对话历史、让模型下一轮真的能看见的结果块
    state 当前运行状态 主循环向下执行时，需要一直带着走的数据

核心角色定义：
    user: 表示用户输入的内容，用户提问、指令或需要模型处理的请求
    assistant: 表示模型的回复内容，记录模型的历史回复，用于维持对话连贯性
    system: 定义对话的背景、规则或模型的初始设定，设置模型行为、风格或全局约束条件

以下是一个简单的最小心智模型
    user message
        |
        v
        LLM
           |
           +-- 普通回答 ----------> 结束
           |
           +-- tool_use ----------> 执行工具
                                      |
                                      v
                                 tool_result
                                      |
                                      v
                                 写回 messages
                                      |
                                      v
                                 下一轮继续
这里真正关键的点是 工具结果必须重新写入进历史消息里面，成为下一轮推理的输入，如果少了这一步，模型就会缺少上下文信息，导致判断有误
"""

#---------------------------关键数据结构-------------------------------

# message 消息 模型下一轮工作需要的上下文
"""
示例：
    {"role": "user", "content": "..."}
    {"role": "assistant", "content": [...]}
"""

# tool result block
# 工具执行完成之后，将信息流包装回去
"""
{
    "type": "tool_result",
    "tool_use_id": "...", tool_use_id 这个参数就是告诉模型这条结果对应的是刚才哪一次的工具调用
    "content": "...",
}
"""

# LoopState 循环状态
"""
state = {
    "messages": [...],
    "turn_count": 1,
    "transition_reason": None,  这个字段主要标识，这一轮结束后，为什么要继续下一轮。
}
"""

#----------------------------最小实现-------------------------------------

# STEP 1 准备初始消息
# 用户请求先进入messages
"""
messages = [{"role": "user", "content": query}]
"""

# STEP 2 模型调用
# 把消息历史、system prompt 和工具定义一起发给模型：
"""
response = client.messages.create(
    model=MODEL,
    system=SYSTEM,
    messages=messages,
    tools=TOOLS,
    max_tokens=8000,
)
"""

# STEP 3 追加 assistant 回复
# 保证模型输出的内容也在上下文中，保证模型上下文不中断
"""
messages.append({"role": "assistant", "content": response.content})
"""

# STEP 4 判断模型是否调用了工具 如果模型调用了工具则执行
"""
results = []
for block in response.content:
    if block.type == "tool_use":
        output = run_bash(block.input["command"])
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        })
"""

# STEP 5 工具结果作为新消息写回上下文，注意这里将结果放回上下文的角色要是user，也算是作为用户的提问输入
"""
messages.append({"role": "user", "content": results})
"""

#-----------------------------完整的最小版本agent loop----------------------------------
"""
def agent_loop(state):
    while True:
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM,
            messages=state["messages"],
            tools=TOOLS,
            max_tokens=8000,
        )

        state["messages"].append({
            "role": "assistant",
            "content": response.content,
        })

        if response.stop_reason != "tool_use":
            state["transition_reason"] = None
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_tool(block)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

        state["messages"].append({"role": "user", "content": results})
        state["turn_count"] += 1
        state["transition_reason"] = "tool_result"
"""

#----------------------------完整代码实现-------------------------------------
import os
import subprocess
from dataclasses import dataclass
import json

#增强终端输入体验的主要功能，这是为了修复 macOS 下 libedit 环境中 UTF-8 输入时退格键异常 的问题。
try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
    readline.parse_and_bind('set enable-meta-keybindings on')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

# 获取环境变量配置文件
load_dotenv(override=True)

# 如果没有自定义的ANTHROPIC_BASE_URL，就主动清理掉ANTHROPIC_AUTH_TOKEN，防止认证冲突
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# 获取GPT client
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))

# 获取模型ID
MODEL = os.getenv("MODEL_ID")

# system角色提示词
SYSTEM = (
    f"你是一个在当前工作目录 {os.getcwd()} 中运行的代码助手。使用 bash 命令查看并修改工作区。先执行，再清楚说明你做了什么。"
)

# 定义gpt可调用的工具 TOOLS
TOOLS = [
    {
        "name": "bash",
        "description": "在当前工作区执行shell脚本命令",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
    }
]

# 定义循环历史状态类（最小化的循环状态类，包含历史记录、循环数、继续执行原因）
@dataclass
class LoopHistory:
    messages: list
    turn_count: int = 1
    transition_reason: str | None = None

# bash命令执行函数
def run_bash(command: str) -> str:
    # 一个简易的命令黑名单
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]

    if any(item in command for item in dangerous):
        return "Error: 危险命令禁止使用"

    # 使用subprocess工具执行bash命令并保存结果
    try:
        result = subprocess.run(
            command,
            shell=True, # 通过shell执行命令
            cwd=os.getcwd(), # 获取当前工作目录 表示在当前工作目录下执行
            capture_output=True, # 捕获标准输出 stdout 捕获标准错误 stderr 否则命令输出会直接打印到终端，而不是保存到程序里
            text=True, # 输出按照文本字符串处理 不是字节流
            timeout=120, # 执行超时时间
        )
    except subprocess.TimeoutExpired:
        return "Error: 运行超时，当前设定超时时间为120s"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

    # 将标准输出、标准错误拼接在一起 去除前后空格
    output = (result.stdout + result.stderr).strip()

    # 限制返回长度为50000个字符 如果没有任何输出则输出 无输出
    return output[:50000] if output else "(无输出)"

# 文本处理函数
def extract_text(content) -> str:

    # 判断入参content是否是列表 如果不是列表则返回空字符串
    if not isinstance(content, list):
        return ""

    # 定义texts用于存放循环遍历提取的文本
    texts = []

    # 循环遍历content
    for item in content:

        # 判断item中是否有text这个属性 或字段 如果没有返回None
        text = getattr(item, "text", None)
        if text:
            # 如果有值则加入texts列表
            texts.append(text)
    # texts列表中的元素使用换行符拼接 并去除首位空白
    return "\n".join(texts).strip()

# 工具执行函数
def execute_tool_calls(response_content) -> list[dict]:

    # 定义结果列表
    results = []

    # 循环遍历模型返回文本
    for block in response_content:
        # 判断当前元素是否是tool_use类型
        if block.type != "tool_use":
            # 如果不是直接跳过
            continue
        # 从元素中取出input字段，从input字段中取出command 赋给当前的command值
        command = block.input["command"]

        # 在终端中打印出来 使用ANSI转义码显示为黄色
        print(f"\033[33m$ {command}\033[0m")

        # 调用上方定义的run_bash函数执行命令，并将执行结果保存在output中
        output = run_bash(command)

        # 在控制台中打印输出output，限制打印长度为200个字符
        print(output[:200])

        # 结果封装 将结果的类型封装为tool_result返回
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        })
    # 返回结果
    return results

# 大模型调用一个轮次函数
def run_one_turn(history: LoopHistory) -> bool:

    # 调用大模型 保存返回
    responses = client.messages.create(
        model=MODEL, # 指定模型类型
        system=SYSTEM, # 系统提示词
        messages=history.messages, # 完整历史对话
        tools=TOOLS, # 模型可用的工具列表
        max_tokens=8000, # 本次回复最多可生成多少token
    )

    # 打印大模型返回
    print("\n===== responses josn=====")
    print(json.dumps(responses.model_dump(), ensure_ascii=False, indent=2))
    print("==========================\n")

    # 将模型的回复内容加入历史消息 这样后面的下一轮调用模型时，就能看到这段历史大模型回复
    history.messages.append({
        "role": "assistant",
        "content": responses.content,
    })

    # 判断大模型是否需要调用工具
    if responses.stop_reason != "tool_use":
        # 如果此次模型停止原因不是因为要调用工具，将继续执行原因赋值为None
        history.transition_reason = None
        # 则返回false 说明这一轮就到这一步为止，不需要继续执行，对话暂时结束
        return False

    # 上一个判断判断了是否需要调用工具 如果上个判断没有走 走到这一步 就相当于模型需要调用工具执行相关操作
    # 执行工具调用方法 并保存返回结果
    results = execute_tool_calls(responses.content)

    # 如果工具结果为空或者无执行结果，则此轮也无法继续进行，直接结束
    if not results:
        history.transition_reason = None
        return False

    # 上一个判断判断了工具执行是否有结果 如果没有则停止 走到这一步说明有结果
    # 将工具结果作为【用户消息】追加进历史中
    """
    这个地方很关键 虽然是一次工具执行的结果，但是需要将其包装成为一条新的【用户消息】加回去，这是很多tool calling协议里常用的做法
    因为工具结果通常需要作为下一轮输入喂回给模型，而不是作为assistant模型回复喂回给模型
    你可以这么理解： 相当于用户又把模型的执行结果提交给模型，模型再继续推理， 只不过这个“用户”不再是你自己，而是程序。
    """
    history.messages.append({
        "role": "user",
        "content": results
    })

    # 循环继续
    # 历史记录中轮数记录+1
    history.turn_count += 1

    # 历史记录里记录本次继续循环原因是 使用了工具 并且有返回结果 原因设置为"tool_result"
    history.transition_reason = "tool_result"

    # 执行完工具之后 继续执行下一轮 因为模型现在已经拿到工具结果了，通常还要再调用一次模型，让它基于工具结果生成最终回答。
    """
    用一句话总结这段代码:执行一轮“让模型先说话，如果它要用工具就执行工具，并把工具结果塞回上下文，为下一轮继续生成做准备”的流程。
    你可以把它理解成这个流程图:
    用户历史消息  -> 调模型
                -> 模型回复
                -> 是否要调用工具？
                        否 -> 结束，返回 False
                        是 -> 执行工具
                            -> 有结果吗？
                                否 -> 结束，返回 False
                                是 -> 把工具结果加入消息历史
                                    -> 返回 True，准备下一轮
    """
    return True

# 最小心智模型循环 agent的主循环
def agent_loop(history: LoopHistory) -> None:

    # 不停的执行 上述代码中定义的大模型一个轮次函数 直到返回false为止
    # 如果函数返回Ture 则pass什么都不做 继续执行下一轮
    # 如果函数返回False 则循环结束
    while run_one_turn(history):
        pass

# 将整个代码应用起来 做一个命令行里的多轮对话程序
if __name__ == "__main__":

    # 初始化聊天历史 用于保存整个对话历史
    historyMessage = []

    # 进入死循环，不断等待用户输入
    while True:

        try:

            query = input("\033[36m01_angent_loop >> \033[0m")
        # 按键 Ctrl+D 触发EOFError 按键Ctrl+C 触发KeyboardInterrupt
        except (EOFError, KeyboardInterrupt):
            # 触发以上两个按键直接退出循环
            break
        # 判断是否退出 如果输入匹配到 q quit exit 空字符串 直接退出循环
        if query.strip().lower() in ("q", "quit", "exit", ""):
            break

        # 将用户输入加入历史记录
        historyMessage.append({
            "role": "user",
            "content": query
        })

        # 将当前的聊天记录放进循环历史状态中，初始化agent的运行状态
        loop_state = LoopHistory(messages=historyMessage)

        # 执行agent主循环
        agent_loop(loop_state)

        # 最终提取文本 将historyMessage中最后一条的content取出
        final_text = extract_text(historyMessage[-1]["content"])

        # 如果最终提取文本不为空 则打印出来
        if final_text:
            print(final_text)
        # 额外打印一个空行
        print()


"""
总结：
    以上是一个最小的agent的循环流程，提供了大模型调用client、历史会话消息聚合、agent状态管理、交互对话输入输出等功能，实现了一个最小的心智模型
    总结一下实现的功能点就是：
    这段代码就是一个终端聊天程序入口：
    1. 等你输入问题
    2. 把问题放进聊天历史
    3. 调 agent 去处理
    4. 拿到最终回复
    5. 打印出来
    6. 继续等你输入下一句
    
    相当于以下这样一个流程：
        启动程序
          ↓
        等待用户输入
          ↓
        输入 q / exit / 空行？ —— 是 → 退出
          ↓ 否
        保存到 history
          ↓
        调用 agent_loop(loop_state)
          ↓
        agent 跑完整个模型/工具流程
          ↓
        取最后一条消息
          ↓
        提取文本并打印
          ↓
        继续下一轮
"""

