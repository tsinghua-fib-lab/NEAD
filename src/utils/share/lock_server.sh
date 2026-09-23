# Start the server from the repository root with:
# PYTHONPATH=. python src/utils/share/lock_server.py
# Source this file to make the allocate function available.
# Set LOCK_SERVER=http://server-ip:port to coordinate across machines.
# If LOCK_SERVER is empty (the default), every task is accepted locally.
allocate() {
    local server="$1" # Optional: http://server-ip:port
    local taskname="$2"

    if [[ -z "$server" ]]; then
        return 0
    fi

    # Terminal colors
    local bold="\033[1m"
    local green="\033[1;38;5;82m"
    local yellow="\033[1;38;5;220m"
    local red="\033[1;38;5;196m"
    local reset="\033[0m"

    local payload="{\"task\":\"${taskname}\",\"owner\":\"$(hostname)\"}"
    local response=$(
        curl -s --max-time 10 -X POST "$server/claim" -H "Content-Type: application/json" -d "$payload" 2>/dev/null
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
