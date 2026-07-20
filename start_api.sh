#!/bin/zsh
cd /Users/shiaho/Desktop/bitx
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy 2>/dev/null
pkill -f 'anthropic_server.py' 2>/dev/null
sleep 1
mkdir -p kef_results
echo "Base URL 请填: http://127.0.0.1:8787"
echo "不要填:      http://127.0.0.1:8787/v1/messages"
python3 -u anthropic_server.py --host 127.0.0.1 --port 8787 2>&1 | tee kef_results/anthropic_server.log
