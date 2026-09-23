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
        print(f"\n[grey]Timed out after {timeout}s; using default: {default}[/grey]")
        return default
    return result[0]

def set_save_dir(save_dir: str, name: str) -> str:
    console = Console()
    save_dir = Path(save_dir)
    while True:
        if name is None:
            name = input_with_timeout(
                "[bold yellow]--name is missing. Enter a name or press Enter to skip.[/bold yellow]",
                default="debug",
            )
            if name == "debug":
                save_dir = save_dir / name
                console.print(
                    f"[blue]Results will be saved to {save_dir}; a later debug run may overwrite them.[/blue]"
                )
                break

        if (save_dir / name).exists():
            suffix = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            choice = input_with_timeout(
                f"[bold yellow]--name '{name}' already exists. Choose an action:\n"
                "[1] Overwrite the existing results\n"
                f"[2] Save to {name}_{suffix} (default)\n"
                "[3] Enter a new name\n"
                "[/bold yellow]",
                default="2",
            )
            if choice == "1":
                save_dir = save_dir / name
                console.print(
                    f"[blue]Results will be saved to {save_dir}, overwriting existing logs.[/blue]"
                )
                break
            elif choice == "2":
                name = f"{name}_{suffix}"
                save_dir = save_dir / name
                console.print(f"[blue]Results will be saved to {save_dir}.[/blue]")
                break  # The timestamp makes the directory name unique.
            elif choice == "3":
                name = Prompt.ask("[bold yellow]Enter a new name[/bold yellow]").strip()

        save_dir = save_dir / name
        console.print(f"[blue]Results will be saved to {save_dir}.[/blue]")
        break

    return str(save_dir), name
