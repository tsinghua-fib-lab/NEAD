import json
import logging
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from filelock import FileLock
from pydantic import BaseModel
from argparse import ArgumentParser
from src.utils.tag2ansi import tag2ansi
from src.utils.logger import init_logger
from src.utils.share.get_ip import get_local_ip

_logger = logging.getLogger('src')

LOCK_FILE = None # Path("./playground/locks.json")
LOCK_GUARD = None # FileLock(LOCK_FILE.with_suffix('.lock'))
app = FastAPI()

class Task(BaseModel):
    task: str
    owner: str = "unknown"

def read_locks():
    with open(LOCK_FILE, "r") as f:
        return json.load(f)

def write_locks(data):
    with open(LOCK_FILE, "w") as f:
        json.dump(data, f, indent=2)

@app.post("/claim")
def claim(task: Task):
    with LOCK_GUARD:
        locks = read_locks()
        if task.task in locks:
            _logger.info(f"{task.owner} failed to claim {task.task}, already taken by {locks[task.task]['owner']}")
            return {"ok": False, "msg": "already taken"}
        locks[task.task] = {"status": "taken", "owner": task.owner}
        write_locks(locks)
    _logger.info(f"{task.owner} claimed {task.task}")
    return {"ok": True, "msg": f"claimed by {task.owner}"}

@app.post("/done")
def done(task: Task):
    with LOCK_GUARD:
        locks = read_locks()
        if task.task in locks:
            locks[task.task]["status"] = "done"
            write_locks(locks)
    _logger.info(f"{task.owner} marked {task.task} as done")
    return {"ok": True}

@app.get("/status")
def status():
    with LOCK_GUARD:
        return read_locks()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--lock_file", type=str, default='./src/utils/share/lock_server.json', help="Path to the lock file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=16699, help="Port to run the server on")
    args = parser.parse_args()
    LOCK_FILE = Path(args.lock_file)
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not LOCK_FILE.exists():
        with open(LOCK_FILE, "w") as f:
            json.dump({}, f)
    LOCK_GUARD = FileLock(LOCK_FILE.with_suffix('.lock'))
    init_logger('src')
    _logger.info(f"Lock server running on {args.host}:{args.port} with lock file {LOCK_FILE}")
    _logger.note(tag2ansi(
        f"Use [pink]curl -s -X POST http://{get_local_ip()}:{args.port}/claim -d '{{\"task\": \"task_name\", \"owner\": \"owner_name\"}}' -H 'Content-Type: application/json'[reset] to claim a task."
    ))
    _logger.note(tag2ansi(
        f'You may use [pink]sudo iptables -A INPUT -p tcp --dport {args.port} -j ACCEPT[reset] to open the port if needed.'
    ))
    uvicorn.run(app, host=args.host, port=args.port)
