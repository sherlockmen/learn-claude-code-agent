"""
在02小节中agent已经具备了tool能力，但是新的问题也来了，我们虽然在代码中写了tool黑名单列表，
但是并不能完全的防止危险操作，所以本小节我们引入工具permission，权限系统的概念

在工具执行前设置三道关卡：

  第 1 关：硬性禁止列表（如 rm -rf /、sudo 等）
  第 2 关：规则匹配（是否写入工作区之外？是否为破坏性命令？）
  第 3 关：用户审批（暂停执行，等待用户确认）

这种模式下 当前我们的循环流程就变成了以下的流程 用户输入提示词--->大模型--->权限系统--->调用工具--->返回结果
    +----------+      +-------+      +--------------+      +---------------+
    |   User   | ---> |  LLM  | ---> | Permission   | ---> | Tool Dispatch |
    |  prompt  |      |       |      | 1. deny list |      | execute       |
    +----------+      +---+---+      | 2. rules     |      +-------+-------+
                          ^          | 3. approval  |              |
                          |          +------+-------+              |
                          |                 | deny                 |
                          |                 v                      v
                          |          +-------------------------------+
                          +----------+ tool_result: denied or output |
                                     +-------------------------------+

--------------------------拆解工作原理-----------------------------
1. deny list 拒接策略----一个硬拒绝表，先查，命中就返回阻止信息。
DENY_LIST = [
    "rm -rf /", "sudo", "shutdown", "reboot",
    "mkfs", "dd if=", "> /dev/sda",
]

def check_deny_list(command: str) -> str | None:
    for pattern in DENY_LIST:
        if pattern in command:
            return f"Blocked: '{pattern}' 在拒绝列表中，禁止执行"
    return None

2. ask list 规则匹配----描述"什么时候需要问用户"，每条规则指定工具和检查条件。
PERMISSION_RULES = [
    {
        "tools": ["write_file", "edit_file"],
        "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
        "message": "Writing outside workspace",
    },
    {
        "tools": ["bash"],
        "check": lambda args: any(kw in args.get("command", "") for kw in ["rm ", "> /etc/", "chmod 777"]),
        "message": "Potentially destructive command",
    },
]

def check_rules(tool_name: str, args: dict) -> str | None:
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None

3. user ask 用户审批----规则命中后，暂停等待用户审批
def ask_user(tool_name: str, args: dict, reason: str) -> str:
    print(f"\n⚠  {reason}")
    print(f"   Tool: {tool_name}({args})")
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"
"""

# ---------------------------------------完整实现----------------------------------------

import os
import re
import subprocess
from operator import ifloordiv
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

import glob as g

try:
    import readline

    # 避免终端特殊按键配置影响命令行输入
    readline.parse_and_bind('set bind-tty-special-chars off')
    # 支持中文、特殊符号等非英文字符输入
    readline.parse_and_bind('set input-meta on')
    # 支持中文、特殊符号等字符正常显示
    readline.parse_and_bind('set output-meta on')
    # 防止特殊字符被转换成 ESC 快捷键序列
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass

# 获取当前.env文件中的环境变量
load_dotenv(override=True)

# 如果没有自定义的ANTHROPIC_BASE_URL，就主动清理掉ANTHROPIC_AUTH_TOKEN，防止认证冲突
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# 获取当前的工作目录 并储存
WORKDIR = Path.cwd()

# 获取GPT客户端
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))

# 指定使用模型
MODEL = os.environ["MODEL_ID"]

# 系统提示词
SYSTEM = f"你是一个在 {WORKDIR} 工作目录中运行的编程智能体。所有具有破坏性的操作都必须经过用户批准。"

# -------------------------------------------02小节的工具实现------------------------------------------

"""执行shell命令 并返回输出结果"""
def run_bash(command: str) -> str:
    try:

        # 在 WORKDIR 目录下执行命令
        r = subprocess.run(
            command,                # 入参命令
            shell=True,             # 使用 Shell 执行命令，例如 ls、grep、python xxx.py
            cwd=WORKDIR,            # 指定命令执行目录
            capture_output=True,    # 捕获标准输出和错误输出
            text=True,              # 输出结果转成字符串，而不是 bytes
            errors="replace",       # 遇到无法解码的字符时，用替代字符处理
            timeout=120             # 超时时间
        )

        # 合并正常输出 stdout 和错误输出 stderr，并去掉首尾空白
        out = (r.stdout + r.stderr).strip()

        # 最多返回 50000 个字符；没有输出则返回提示
        return out[:50000] if out else "(无输出)"
    except subprocess.TimeoutExpired:

        # 命令执行超过 120 秒
        return "Error: 运行超时，当前设定超时时间为120s"

"""读取文件内容"""
def run_read(path: str, limit: int | None = None) -> str:
    try:
        # 拼接 WORKDIR 和文件路径，得到绝对路径
        # 然后按 UTF-8 读取文件，并按行拆分
        lines = (WORKDIR / path).resolve().read_text(encoding="utf-8").splitlines()

        # 如果设置了 limit，并且文件行数超过 limit
        if limit and limit < len(lines):

            # 只保留前 limit 行，并提示还有多少行没显示
            lines = lines[:limit] + [f"... ({len(lines) - limit} 行未显示)"]

        # 把每一行重新用换行符拼起来
        return "\n".join(lines)
    except Exception as e:

        # 文件不存在、权限不足、编码错误等都会在这里返回
        return f"Error: {e}"

"""创建或覆盖文件"""
def run_write(path: str, content: str) -> str:
    try:
        # 得到目标文件的绝对路径
        file_path = (WORKDIR / path).resolve()

        # 如果父目录不存在，就自动创建
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 使用 UTF-8 写入文件
        # 如果文件已经存在，会直接覆盖
        file_path.write_text(content, encoding="utf-8")

        # 返回写入结果
        return f"写入 {len(content)} bytes 在 {path} 路径中"
    except Exception as e:
        # 写入失败时返回错误
        return f"Error: {e}"

"""替换文件中的指定文本"""
def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        # 获取文件绝对路径
        file_path = (WORKDIR / path).resolve()

        # 读取原文件内容
        text = file_path.read_text(encoding="utf-8")

        # 如果找不到要替换的旧文本，直接返回错误
        if old_text not in text:
            return f"Error: 未找到该文件 {path}"

        # 将第一次出现的 old_text 替换成 new_text
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")

        # 返回修改成功
        return f"已完成修改 {path}"
    except Exception as e:

        # 发生异常则返回错误信息
        return f"Error: {e}"


"""根据通配符查找文件"""
def run_glob(pattern: str) -> str:

    try:
        # 根据 pattern 搜索匹配的文件
        matches = sorted({
            match for match in g.glob(
                pattern,
                root_dir=WORKDIR,       # 从 WORKDIR 开始查找
                recursive=True          # 支持 ** 递归查找
            )

            # 只保留 WORKDIR 目录内部的文件
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR)
        })

        # 最多展示前 200 个结果
        shown = matches[:200]

        # 如果结果超过 200 个，就加一个提示
        if len(matches) > 200:
            shown.append("... (超过 200 条匹配结果，仅显示前 200 条)")

        # 用换行符拼接结果
        # 如果没有匹配结果，返回 未匹配到
        return "\n".join(shown) if shown else "(未匹配到)"
    except Exception as e:

        # 查找异常时返回错误信息
        return f"Error: {e}"

# -------------------------------------------02小节工具分发系统------------------------------------------


"""
工具列表 TOOLS
主要功能是告知模型系统有哪些工具，以及每个工具的输入格式。可以理解成是给模型的工具说明书，告诉模型有哪些工具可调用，工具需要哪些参数，工具是干什么的

以工具列表中的bash工具为例，说明一下这个工具的对象都包含什么
{
	"name": "bash",                            -----> 工具名
	"description": "Run a shell command.",     -----> 工具说明 描述工具是干嘛的
	"input_schema": {                          -----> 输入参数结构定义 
		"type": "object",                      -----> 规定入参的类型 
		"properties": {                        -----> 入参的字段
			"command": {                       -----> 入参字段有一个是 command 
				"type": "string"               -----> 入参种类是字符串
			}
		},
		"required": ["command"]                -----> 必填参数列表
	}
}

"""
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern; ** matches recursively.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
]

"""
TOOL_HANDLERS 工具分发映射表 
本质上这个映射表就是一个字典，key是工具名，value是对应的处理函数
lambda **params 的意思就是定义了一个匿名函数，接收关键字参数
"""
TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}

# -------------------------------------------03小节权限系统具体实现------------------------------------------

#Agent 调用工具 → Gate 1：黑名单检查 → Gate 2：危险规则检查 → Gate 3：如果有风险，询问用户是否允许 → 允许 / 拒绝执行

# =========================
# Gate 1：命令黑名单
# =========================

# 明确禁止执行的高危命令
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"]

"""检查 Shell 命令是否命中禁止执行的黑名单"""
def check_deny_list(command: str) -> str | None:

    # 遍历所有禁止的命令模式
    for deny in DENY_LIST:

        # 只要命令中包含黑名单内容，就直接拦截
        if deny in command:
            return f"{deny}该命令为高危命令 拒绝执行"

    # 没有命中黑名单
    return None

# =========================
# Gate 2：危险操作规则检查
# =========================

# 匹配 rm / del 删除命令
# (?i) 表示忽略大小写
# 主要避免把字符串中普通的 "rm"、"del" 误识别成命令
DESTRUCTIVE_COMMAND_WORD = re.compile(
    r"(?i)(?:^|[;&|()\n])\s*(?:rm|del)(?=\s|$|[;&|()])"
)

"""判断命令中是否包含 rm / del 等删除操作"""
def contains_destructive_command(command: str) -> bool:

    # 匹配到返回 True，否则返回 False
    return bool(DESTRUCTIVE_COMMAND_WORD.search(command))

# 工具权限规则
PERMISSION_RULES = [

    # 文件操作规则
    {
        "tools": ["read_file", "write_file", "edit_file"],

        # 判断访问路径是否超出 WORKDIR 工作目录
        "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
        "message": "访问路径超出工作目录"
    },

    # Shell 命令规则
    {
        "tools": ["bash"],

        # 以下情况认为是危险操作：
        # 1. 包含 rm / del
        # 2. 包含 rm
        # 3. 向 /etc/ 写文件
        # 4. chmod 777 修改成完全开放权限
        "check": lambda args: contains_destructive_command(args.get("command", "")) or
        any(kw in args.get("command", "") for kw in ["rm ", "> /etc/", "chmod 777"]),
        "message": "检测到潜在危险命令"
    }
]

"""检查某个工具调用是否命中危险操作规则"""
def check_rules(tool_name:str, args: dict) -> str | None:

    # 遍历所有权限规则
    for rule in PERMISSION_RULES:

        # 当前工具属于该规则，并且规则检查结果为 True
        if tool_name in rule["tools"] and rule["check"](args):

            # 返回触发规则的原因
            return rule["message"]

    # 没有命中任何危险规则
    return None


# =========================
# Gate 3：用户确认
# =========================

"""危险操作执行前，询问用户是否允许"""
def ask_user(tool_name: str, args: dict, reason: str) -> str:
    #黄色显示风险提示
    print(f"\n\033[33m[permission] {reason}\033[0m")

    # 显示 Agent 准备调用的工具和参数
    print(f"   Tool: {tool_name}({args})")

    # 等待用户输入 y / yes
    choice = input("   Allow? [y/N] ").strip().lower()

    # y / yes 表示允许，其他输入默认拒绝
    return "allow" if choice in ("y", "yes") else "deny"

# =========================
# 三道权限检查统一入口
# =========================

"""检查 Agent 的一次工具调用是否允许执行"""
def check_permission(block) -> bool:

    # 第一层：只有 bash 命令需要检查命令黑名单
    if block.name == "bash":

        # 获取 Shell 命令并检查黑名单
        reason = check_deny_list(block.input.get("command", ""))

        # 命中黑名单：直接禁止，不询问用户
        if reason:
            print(f"\n\033[31m[blocked] {reason}\033[0m")
            return False

    # 第二层：检查普通权限规则
    reason = check_rules(block.name, block.input)

    # 命中危险规则
    if reason:

        # 第三层：询问用户是否允许执行
        decision = ask_user(block.name, block.input, reason)

        # 用户拒绝
        if decision == "deny":
            return False

    # 没有风险，或者用户已经批准
    return True

# -------------------------------------------03小节 agent loop 添加了权限系统------------------------------------------

"""Agent 主循环"""
def agent_loop(message: list):

    # 持续运行，直到模型不再调用工具
    while True:
        # 调用大模型
        response = client.messages.create(
            model=MODEL,        # 使用的模型
            system=SYSTEM,      # 系统提示词
            messages=message,   # 当前完整对话上下文
            tools=TOOLS,        # 提供给模型使用的工具列表
            max_tokens=8000     # 单次最大输出 Token
        )

        # 把模型本轮回复加入对话历史
        message.append({
            "role": "assistant",
            "content": response.content
        })

        # 从模型回复中提取所有工具调用请求
        tool_calls = [
            block for block in response.content if block.type == "tool_use"
        ]

        # 如果模型没有调用工具，说明任务已经完成，退出循环
        if not tool_calls:
            return

        # 保存本轮所有工具的执行结果
        results = []

        # 依次处理模型请求的每一个工具
        for block in tool_calls:

            # 打印当前准备执行的工具名称
            print(f"\033[36m> {block.name}\033[0m")

            # 执行工具前先进行权限检查
            if not check_permission(block):

                # 权限检查未通过，告诉模型该工具被拒绝
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "权限拒绝"
                    }
                )
                # 跳过当前工具，继续处理下一个
                continue

            # 根据工具名称找到对应的 Python 处理函数
            handler = TOOL_HANDLERS.get(block.name)

            # 找到处理函数就执行，并传入模型生成的参数
            # 如果没有找到，则返回 未知的工具
            output = handler(**block.input) if handler else  f"未知的工具 {block.name}"

            # 在终端中打印工具执行结果的前 200 个字符
            print(str(output)[:200])

            # 把工具执行结果保存起来
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output
            })

        # 把所有工具执行结果作为 user 消息返回给模型
        # 下一轮模型会根据这些结果继续分析和执行
        message.append({
            "role": "user",
            "content": results
        })



if __name__ == "__main__":
    print("✅03: 权限系统")
    print("输入你想问的问题，按回车发送，输入q退出\n")

    # 保存整个对话历史
    history = []
    # 持续等待用户输入
    while True:
        try:
            # 显示命令行输入提示符：✅03: 权限系统 >>
            query = input("\033[36m✅03: 权限系统 >> \033[0m")

        # Ctrl+D 或 Ctrl+C 时退出程序
        except (EOFError, KeyboardInterrupt):
            break
        # 输入 q、exit 或空内容时退出
        if query.strip().lower() in ("q", "exit", ""):
            break

        # 把用户输入加入对话历史
        history.append({"role": "user", "content": query})

        # 调用 Agent 主循环处理用户请求
        agent_loop(history)

        # 获取 Agent 最后一条回复内容
        for block in history[-1]["content"]:

            # 只打印文本类型的内容
            if getattr(block, "type", None) == "text":
                print(block.text)

        # 输出一个空行，让下一轮显示更清晰
        print()

"""
总结：以上是加入了权限系统的最小agent循环 现在的整体框架变成了以下的一个流程：

用户输入
   ↓
Agent 主循环
   ↓
调用大模型
   ↓
模型决定：
   ├─ 直接回答
   └─ 调用工具
        ↓
     权限检查
        ↓
   ┌────┴────┐
   │         │
允许执行    拒绝执行
   │         │
   ↓         ↓
执行工具   返回 Permission denied
   │
   ↓
把工具结果返回给模型
   ↓
继续下一轮

举一个完整的例子来说明现在的操作流程：
用户输入：帮我修改 main.py 里的 Bug，然后运行测试
agent的工作流程如下：
① 用户请求
     ↓
② agent_loop 调模型
     ↓
③ 模型决定先找文件
     ↓
   glob("**/*.py")
     ↓
④ 找到 main.py
     ↓
   read_file("main.py")
     ↓
⑤ 模型分析代码
     ↓
   edit_file(...)
     ↓
⑥ 权限检查
   普通 WORKDIR 内修改 → 允许
     ↓
⑦ 修改代码
     ↓
⑧ 模型决定运行测试
     ↓
   bash("pytest")
     ↓
⑨ 权限检查
   非危险命令 → 允许
     ↓
⑩ 返回测试结果
     ↓
⑪ 模型发现测试通过
     ↓
⑫ 输出最终答案
     ↓
没有 tool_use
     ↓
agent_loop 结束

"""