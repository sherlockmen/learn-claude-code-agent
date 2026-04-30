"""
这一小节你可以学习到以下的几个功能：
1. 如何构建dispatch map（工具分发表）--- 将工具名和所对应的函数抽象成一个map对象，agent需要使用工具时就去这个表里寻找需要使用的工具。
2. 路径沙箱，如何阻止模型逃离工作目录 --- 防止模型读写到工作区之外的文件
3. 如何在不修改 agent loop的前提下扩展新工具 --- 增加新工具时，不需要重写主循环

----------------------------问题------------------------------
在上一小节的最小agent loop中我们使用到了bash工具，如果这个agent只有这一个bash工具，那么它的所有操作都要通过shell去做，这样会带来以下几个明显的问题：
1. 无法限制agent读取什么文件
2. 无法限制agent将结果写入哪里
3. 无法控制输出大小
4. 任何一个错误的命令就可能破坏文件、泄露敏感数据，或者输出过长把上下文塞爆

--------------------------如何解决-----------------------------
在这种问题背景下，我们引出工具分发表dispatch map这个概念，将原本我们不论什么操作都走bash脚本这种方式转变为准备一组职责明确的专用工具：
例如：读取文件 ---> read_file; 写文件 ---> write_file; 编辑文件 ---> edit_file
每个工具只负责一类事情，并且各自带有对应的安全检测，保证模型不会读取或者写入到其他位置，或无控制的输出上下文。

并且在这种情况下，如果我们后期要对工具进行扩展，我们可以直接只修改dispatch map，不用对agent loop这个主循环进行修改

这种模式下 当前我们的循环流程就变成了以下的流程 用户输入提示词--->大模型--->调用工具--->返回结果
+--------+      +-------+      +------------------+
|  User  | ---> |  LLM  | ---> | Tool Dispatch    |
| prompt |      |       |      | {                |
+--------+      +---+---+      |   bash: run_bash |
                    ^          |   read: run_read |
                    |          |   write: run_wr  |
                    +----------+   edit: run_edit |
                   tool_result | }                |
                               +------------------+

--------------------------拆解工作原理-----------------------------
1. 工具分发表 dispatch map 可以看出 其实这个dispatch map本质上就是一个key是工具名 value是对应函数的一个map表

TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

2. 每个工具都有独立的执行函数，同时在函数中有对应的安全检测如工作目录、输出最大token等，防止大模型在使用工具时逃逸出沙箱工作区，导致文件损坏丢失等不可控情况发生

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_read(path: str, limit: int = None) -> str:
    text = safe_path(path).read_text()
    lines = text.splitlines()
    if limit and limit < len(lines):
        lines = lines[:limit]
    return "\n".join(lines)[:50000]

3. 在循环中按照名称查找处理函数，循环本身与01的循环完全一致


for block in response.content:
    if block.type == "tool_use":
        handler = TOOL_HANDLERS.get(block.name)
        output = handler(**block.input) if handler \
            else f"Unknown tool: {block.name}"
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        })
"""

# ---------------------------------------完整实现----------------------------------------

import os
import subprocess
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

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

# SYSTEM系统提示词
SYSTEM = f"你是一个位于 {WORKDIR} 的编程智能体。使用工具来解决任务。直接行动，不要解释。"

# -------------------------------------------安全防护------------------------------------------
# 路径沙箱函数 --- 把用户传进来的路径做安全校验，防止访问工作目录之外的文件。
def safe_path(p: str) -> Path:

    # (WORKDIR / p)表示拼接路径 例如 WORKDIR = "/app/workspace"， p = "test/a.txt"， (WORKDIR / p) ---> "/app/workspace/test/a.txt"
    # resolve()方法是将路径变成绝对路径
    # 例如 WORKDIR = /app/workspace，p = "../secret.txt"
    # 那么 (WORKDIR / p) ---> /app/workspace/../secret.txt
    # 使用resolve()之后 这个文件目录就变为 先进入/app ---> 再进入/workspace ---> ../返回上一级 ---> 从当前的/workspace目录又回到了/app目录下 ---> /secret.txt 意为在/app这个目录下找/secret.txt这个文件
    # 所以使用resolve()之后
    # 当前的路径 WORKDIR = /app/workspace，p = "../secret.txt" ---> (WORKDIR / p) ---> /app/workspace/../secret.txt ---> resolve() ---> /app/secret.txt
    path = (WORKDIR / p).resolve()

    # 判断当前路径是否在工作路径内
    if not path.is_relative_to(WORKDIR):
        # 如果不在抛出异常
        raise ValueError(f"当前路径不在工作路径内: {p}")
    # 如果在工作路径内 正常返回当前路径
    return path


# -------------------------------------------工具函数------------------------------------------
# bash命令执行函数（与01节一样，无变化）
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

# 写入文件工具函数 入参path 写入文件路径，content 写进的文本内容
def run_write_tool(path: str, content: str) -> str:
    try:

        # 调用路径沙箱函数 判断当前路径是否在工作路径内
        file_path = safe_path(path)

        # 如果目标文件所在的文件夹不存在，就先创建文件夹
        # parents=True 如果上级目录不存在，也一起创建
        # exist_ok=True 如果目录已经存在，不抛出错误
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 写文件
        file_path.write_text(content)
        return f"文件路径 {path} 下写入了 {len(content)} 字节文件"
    except Exception as e:
        # 捕获异常并返回
        return f"Error: {e}"

# 读取文件工具函数 入参path 读取文件路径；limit 限制读取多少行 默认为None不限制
def run_read_tool(path: str, limit: int = None) -> str:
    try:

        # safe_path(path)调用路径沙箱函数 判断当前路径是否在工作路径内
        # .read_text()将文件内容按文本方式读取出来，赋值给text
        text = safe_path(path).read_text()

        # 将整个文本按行拆分变成列表
        lines = text.splitlines()

        # 如果入参传入了limit限制读取行数值，则判断入参limit是否小于lines读取的总行数
        if limit and limit < len(lines):

            # 如果超过了入参limit限制行数，则取前limit行，然后在最后补充提示信息，还有多少行没显示
            lines = lines[:limit] + [f"... (还有 {len(lines) - limit} 未显示)"]

        # 只保留前50000个字符 防止返回内容过长
        return "\n".join(lines)[:50000]
    except Exception as e:

        # 捕获异常并返回
        return f"Error: {e}"

# 编辑文件工具函数 入参path 文件路径 old_text 老文件 new_text 新文件
# 打开一个文件，把里面指定的一段旧文本替换成新文本，只替换第一次出现的位置
def run_edit_tool(path: str, old_text: str, new_text: str) -> str:
    try:

        # 调用沙箱路径函数，判读当前路径是否在工作路径中
        file_path = safe_path(path)

        # 读取全部内容并存放进content中
        content = file_path.read_text()

        # 判断文本内容，看是否有要替换的旧文本
        if old_text not in content:

            # 如果当前文本中没有要替换的旧文本内容就返回一段提示
            return f"Error: 在路径 {path} 中没有找到要替换的内容"

        # 如果找到了 在content中将old_text 替换成 new_text，并且只替换一次
        # 将替换后的新内容重新写回文件
        file_path.write_text(content.replace(old_text, new_text, 1))

        # 返回文件已修改
        return f"在 {path} 路径中的文件已修改"
    except Exception as e:

        # 捕获异常并返回
        return f"Error: {e}"

# -------------------------------------------工具并发安全分类------------------------------------------
# 对工具做并发安全分类，目的是决定：哪些工具可以同时执行，哪些工具必须一个一个串行执行。⚙
# 并发安全工具
CONCURRENCY_SAFE = {"read_file_tool"}

# 并发不安全工具
CONCURRENCY_UNSAFE = {"write_file_tool", "edit_file_tool"}

# -------------------------------------------工具分发系统------------------------------------------
# 这个系统分两个部分，一部分负责 “工具名 -> 真正执行函数” 的映射，另一部分负责 “告诉模型/程序有哪些工具可用，以及每个工具需要什么参数”。
# 用一句话解释就是 它相当于在做两件事：1.登记工具怎么执行 2.登记工具长什么样、要传什么参数
# PART 1 TOOL_HANDLERS 工具分发映射表（dispatch map）
"""
TOOL_HANDLERS 工具分发映射表 
本质上这个映射表就是一个字典，key是工具名，value是对应的处理函数
lambda **params 的意思就是定义了一个匿名函数，接收关键字参数
"""
TOOL_HANDLERS = {
    "bash": lambda **params: run_bash(params["command"]),
    "read_tool": lambda **params: run_read_tool(params["path"], params["content"]),
    "write_tool": lambda **params: run_write_tool(params["path"], params["content"]),
    "edit_tool": lambda **params: run_edit_tool(params["path"], params["old_text"], params["new_text"]),
}

# PART 2 工具列表TOOLS
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
    {"name": "write_file", "description": "Write content to file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
]

# -------------------------------------------标准化消息格式------------------------------------------
"""
在大模型的历史会话中会存在以下的几种问题：
1. 历史消息中混入大模型"API不能解析或不兼容的字段"，导致API报错
2. 解决tool_use没有对应返回tool_result的问题
3. 解决连续相同角色消息的问题，有些模型API可能不支持连续相同角色交互
基于以上的问题场景，我们引入了标准化消息格式方法的概念，通过一个函数方法将上述的问题包装成一个函数，用来解决大模型历史消息不标准导致的接口报错问题
它解决的不是业务逻辑问题，而是 Agent 调用大模型 API 时的消息格式问题。尤其是在带工具调用的 Agent 里，messages 很容易变乱，所以需要这个函数统一清洗。

总结为一句话即为：把历史对话 messages 整理成大模型 API 能正常接收的格式，避免因为消息格式不规范导致接口报错。

问题1：解决消息里混入“API 不认识字段”的问题
样例JSON：
{
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "我准备调用工具",
            "_debug": "调试信息",        #--------> 自定义字段
            "_cost": 123                #--------> 自定义字段
        }
    ]
}

在上述的JSON串种有两个自定义的字段_debug、_cost这种字段可能是你对接的业务系统程序内部消费的字段或提供的字段，在模型的API中是无法消费和解析的，所以我们需要将这种无法解析和消费的字段剔除
只保留大模型API可解析可消费的字段
{
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "我准备调用工具"
        }
    ]
}

问题2： 解决 tool_use 没有对应 tool_result 的问题 
在agent循环的过程中 每调用一次工具，都需要得到对应的调用工具的结果，例如以下样例
样例JSON：这段JSON是一个assistant 请求调用工具，工具调用 ID 是 tool_001
{
    "role": "assistant",
    "content": [
        {
            "type": "tool_use",
            "id": "tool_001",
            "name": "run_bash",
            "input": {
                "command": "ls"
            }
        }
    ]
}
正常情况下 这个请求过后后面必须有一个对应的工具结果返回
{
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": "tool_001",
            "content": "main.py\nREADME.md"
        }
    ]
}
非正常情况下，模型会遇到执行工具之后工具报错了，导致没有返回结果或者执行完工具之后，返回的结果为空或其他异常情况。这种情况下都会导致接口异常报错
所以我们需要处理这种异常的问题
这个工具的作用就是以下的这个样例情况：
假设你的历史消息是这样：
messages = [
    {
        "role": "user",
        "content": "帮我看看当前目录有什么文件"
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "tool_001",
                "name": "run_bash",
                "input": {
                    "command": "ls"
                }
            }
        ]
    }
]
这表示：
1. 用户问：“当前目录有什么文件？”
2. assistant 说：“我要调用工具执行 ls”
3. 但是后面没有工具执行结果
这个函数工具会自动补一个执行的工具结果上去
{
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": "tool_001",
            "content": "(cancelled)"
        }
    ]
}
意思是：这个工具调用没有结果，或者已经取消了。
这样处理前的历史消息就变成了以下的样子
messages = [
    {
        "role": "user",
        "content": "帮我看看当前目录有什么文件"
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "tool_001",
                "name": "run_bash",
                "input": {
                    "command": "ls"
                }
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tool_001",
                "content": "(cancelled)"
            }
        ]
    }
]
这样就解决了assistant 发起了工具调用，但消息历史里没有工具执行结果，导致 API 报错的情况

问题3 解决连续相同角色消息的问题
有些大模型 API 对消息顺序要求比较严格，通常希望是这种结构：user --> assistant --> user --> assistant 这种用户和助手交替出现的情况
但是在程序运行的过程中很有可能连续的用户输入出现以下的情况
messages = [
    {
        "role": "user",
        "content": "你好"
    },
    {
        "role": "user",
        "content": "帮我写一段 Python 代码"
    }
]
或者这样的回复
messages = [
    {
        "role": "assistant",
        "content": "好的"
    },
    {
        "role": "assistant",
        "content": "我来帮你写"
    }
]
这种连续相同角色的消息有些API可能是不支持的 所以我们需要将这种消息做相应的处理
这个函数工具的解决方案就是将相同的角色消息进行合并处理
处理之前：
messages = [
    {
        "role": "user",
        "content": "你好"
    },
    {
        "role": "user",
        "content": "帮我写一段 Python 代码"
    }
]
处理之后：
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "你好"
            },
            {
                "type": "text",
                "text": "帮我写一段 Python 代码"
            }
        ]
    }
]
将两个消息整合成一个消息，做了合并处理

最后用一个完整的例子来看一下这个工具函数都做了什么操作
假设原始消息为：
messages = [
    {
        "role": "user",
        "content": "帮我查看当前目录"
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "我来调用工具查看一下",
                "_debug": "这里是内部调试信息"
            },
            {
                "type": "tool_use",
                "id": "tool_001",
                "name": "run_bash",
                "input": {
                    "command": "ls"
                },
                "_internal": "内部字段"
            }
        ]
    },
    {
        "role": "user",
        "content": "再帮我看看有没有 Python 文件"
    },
    {
        "role": "user",
        "content": "重点看 .py 后缀的文件"
    }
]
这条消息中显然有以下这几种问题：
1. 有内部字段 _debug、_internal 
2. 工具调用没结果 有tool_use，但是没有tool_result
3. 连续user消息最后两条都是user

经过消息处理工具函数之后会变成以下这种样子：
[
    {
        "role": "user",
        "content": "帮我查看当前目录"
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "我来调用工具查看一下"
            },
            {
                "type": "tool_use",
                "id": "tool_001",
                "name": "run_bash",
                "input": {
                    "command": "ls"
                }
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "再帮我看看有没有 Python 文件"
            },
            {
                "type": "text",
                "text": "重点看 .py 后缀的文件"
            },
            {
                "type": "tool_result",
                "tool_use_id": "tool_001",
                "content": "(cancelled)"
            }
        ]
    }
]
"""
def format_message(message: list) -> list:
    # 定义新列表cleaned 用于存放处理后的消息
    cleaned = []

    # 循环遍历消息列表
    for item in message:
        # 将消息中的角色对应的值取出 赋给clean对象中的role字段
        clean = {"role": item["role"]}
        # 判断消息列表中元素的content字段是不是字符串
        if isinstance(item.get("content"), str):
            # 如果是字符串 则将当前消息列表中的content字段内容赋给clean对象中的content字段
            clean["content"] = item["content"]
        elif isinstance(item.get("content"), list):

            # 保留 block 里所有不是 _ 开头的字段
            clean["content"] = [
                {key: val for key, val in block.items()
                if not key.startswith("_")}
                for block in item["content"]
                if isinstance(block, dict)
            ]
        else:
            # 其余情况做兜底处理 None赋None值 如果没有content字段则将clean对象中的content字段赋值为""
            clean["content"] = item.get("content","")
        # 将处理后的消息添加进新消息列表中
        cleaned.append(clean)

        # 收集消息列表中已经存在的tool_result
        # 定义一个set集合来存储工具结果
        existing_results = set()

        # 循环遍历处理过未知字段的消息列表
        # 获取工具使用id
        for msg in cleaned:
            if isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        existing_results.add(block.get("tool_use_id"))

        # 给缺失的tool_result补齐占位结果
        for msg in cleaned:
            # 如果cleaned消息列表中当前元素的角色不是assistant 或者content的类型不是list
            if msg["role"] != "assistant" or not isinstance(msg["content"], list):
                # 跳过 不进行处理
                continue
            # 如果msg角色不是assistant并且content类型是一个list则走以下逻辑
            # 循环遍历msg中的content元素
            for block in msg["content"]:
                # 如果content中的元素不是一个 dict字典 则跳过不做处理
                if not isinstance(block, dict):
                    continue
                # 如果content当前元素中的type是tool_use 并且id不存在于已经使用过的工具id set集合中
                if block.get("type") == "tool_use" and block.get("id") not in existing_results:
                    # 则说明当前调用的工具没有返回结果 则补充一个占位结果
                    cleaned.append({
                        "role": "user",
                        "content": [
                            {"type": "tool_result",
                             "tool_use_id": block["id"],
                             "content": "(cancelled)"
                             }
                        ]
                    })

        # 合并同角色消息
        # 如果清理后的消息列表是空的 则直接返回
        if not cleaned:
            return cleaned

        # 定义一个新的整合消息列表, 先把第一条消息放进去
        merged = [cleaned[0]]

        # 循环遍历后续消息
        for msg in cleaned[1:]:

            # 判断当前消息和上一条消息是不是同一个角色
            if msg["role"] == merged[-1]["role"]:

                prev = merged[-1]
                prev_content = prev["content"] if isinstance(prev["content"], list) \
                    else [
                    {
                        "type": "text",
                        "text": str(prev["content"])
                    }
                ]
                curr_content = msg["content"] if isinstance(msg["content"], list) \
                    else [
                    {
                        "type": "text",
                        "text": str(msg["content"])
                    }
                ]
                prev["content"] = prev_content + curr_content
            else:
                merged.append(msg)
        return merged


