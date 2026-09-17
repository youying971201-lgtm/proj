#!/bin/zsh
set -eu
unsetopt BG_NICE

cd "$(dirname "$0")"

if ! command -v python3.12 >/dev/null 2>&1; then
    echo "ERROR: 找不到 python3.12。請先安裝 Python 3.12.x。" >&2
    exit 1
fi
python_version="$(python3.12 --version 2>&1)"
case "$python_version" in
    "Python 3.12."*) echo "使用 $python_version" ;;
    *) echo "ERROR: 必須使用 Python 3.12.x，目前是 $python_version" >&2; exit 1 ;;
esac

for arg in "$@"; do
    case "$arg" in
        -h|--help|--list-cameras) exec python3.12 python/main.py "$@" ;;
    esac
done

port=8000
previous_arg=""
for arg in "$@"; do
    if [[ "$previous_arg" == "--port" ]]; then
        port="$arg"
        break
    fi
    case "$arg" in
        --port=*) port="${arg#--port=}"; break ;;
    esac
    previous_arg="$arg"
done

if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "ERROR: 埠號 $port 已被占用，請關閉既有服務或指定其他 --port。" >&2
    exit 1
fi

(
    for ((attempt = 0; attempt < 40; attempt++)); do
        if curl -fs --max-time 1 -o /dev/null "http://127.0.0.1:$port"; then
            open "http://127.0.0.1:$port"
            exit 0
        fi
        sleep 0.5
    done
    echo "ERROR: 伺服器未在 20 秒內啟動。" >&2
) &
opener_pid=$!
trap 'kill "$opener_pid" 2>/dev/null || true' EXIT

python3.12 python/main.py "$@"
