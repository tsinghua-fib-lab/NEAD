# 通过 python ./share/lock_server.py 启动服务器
# 然后 source ./share/lock_server.sh 引用 allocate
# 然后 allocate "http://server-ip:port" "task_name" 来申请任务锁
allocate() {
    local server="$1" # http://server-ip:port
    local taskname="$2"

    # 🎨 颜色定义
    local bold="\033[1m"
    local green="\033[1;38;5;82m"
    local yellow="\033[1;38;5;220m"
    local red="\033[1;38;5;196m"
    local reset="\033[0m"

    local payload="{\"task\":\"${taskname}\",\"owner\":\"$(hostname)\"}"
    local response=$(
        curl -s --max-time 10 -X POST "$LOCK_SERVER/claim" -H "Content-Type: application/json" -d "$payload" 2>/dev/null
    )
    if [[ $? -ne 0 || -z "$response" ]]; then
        echo -e "${bold}${red}[ERROR] Failed to reach lock server at ${server}.${reset}"
        return 1
    elif echo "$response" | grep -q '"ok": *true'; then
        echo -e "${bold}${green}[CLAIMED] ${taskname} has been taken.${reset}"
        return 0
    else
        echo -e "${bold}${yellow}[SKIPPED] ${taskname} already been taken.${reset}"
        return 1
    fi
}
