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


        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"
