import time
import threading
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt


def input_with_timeout(prompt, default="", timeout=30):
    console = Console()
    result = [default]
    def ask():
        try:
            console.print(prompt, end='')
            result[0] = input().strip() or default
        except Exception:
            pass

    thread = threading.Thread(target=ask)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        print(f"\n[grey]超时 {timeout}s，使用默认值: {default}[/grey]")
        return default
    return result[0]

def set_save_dir(save_dir: str, name: str) -> str:
    console = Console()
    save_dir = Path(save_dir)
    while True:
        if name is None:
            name = input_with_timeout(
                "[bold yellow]--name 未指定! 请输入 name, 或按回车跳过[/bold yellow]",
                default="debug",
            )
            if name == "debug":
                save_dir = save_dir / name
                console.print(
                    f"[blue]运行结果将保存到: {save_dir}, 注意, 本次运行结果可能被下次运行覆盖 [/blue]"
                )
                break

        if (save_dir / name).exists():
            suffix = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            choice = input_with_timeout(
                f"[bold yellow]--name '{name}' 已存在, 选择操作: \n"
                "[1] 覆盖已有结果\n"
                f"[2] 保存到 {name}_{suffix} (默认)\n"
                "[3] 自己指定新的 name\n"
                "[/bold yellow]",
                default="2",
            )
            if choice == "1":
                save_dir = save_dir / name
                console.print(
                    f"[blue]运行结果将保存到: {save_dir}, 注意, 这将覆盖已有日志[/blue]"
                )
                break
            elif choice == "2":
                name = f"{name}_{suffix}"
                save_dir = save_dir / name
                console.print(f"[blue]运行结果将保存到: {save_dir}[/blue]")
                break  # 时间戳确保目录不存在，退出循环
            elif choice == "3":
                name = Prompt.ask("[bold yellow]请输入新的 name[/bold yellow]").strip()

        save_dir = save_dir / name
        console.print(f"[blue]运行结果将保存到: {save_dir}[/blue]")
        break

    return str(save_dir), name
