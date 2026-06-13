import requests
import time
import random
import threading
import os
import numpy as np
import lightgbm as lgb
from collections import defaultdict
from flask import Flask
import warnings

warnings.filterwarnings("ignore")

# ================= 配置区 =================
DATA_URL = "https://super.pc28998.com/history/JND28"
TELEGRAM_BOT_TOKEN = "8790154521:AAEUz-Idju8kOEjhqyV9IMv2PEr2ditTUQg"
TELEGRAM_CHAT_ID = "6824519270"
# =========================================

GROUP_MAP = {"大双": 0, "大单": 1, "小双": 2, "小单": 3}
REV_MAP = {0: "大双", 1: "大单", 2: "小双", 3: "小单"}

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload, timeout=5)
    except: pass 

def get_group_type(total_sum):
    is_big = total_sum >= 14
    is_even = total_sum % 2 == 0
    if is_big and is_even: return "大双"
    elif is_big and not is_even: return "大单"
    elif not is_big and is_even: return "小双"
    else: return "小单"

def fetch_api_data(limit=50):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(f"{DATA_URL}?limit={limit}", headers=headers, timeout=10)
        return response.json() if response.status_code == 200 else None
    except: return None

def extract_history(data_json):
    records = []
    if isinstance(data_json, list): records = data_json
    elif isinstance(data_json, dict):
        for key in ['data', 'list', 'result', 'records']:
            if key in data_json and isinstance(data_json[key], list):
                records = data_json[key]
                break
    if not records: return None, None, None
    latest_issue = records[0].get('expect', '未知')
    groups, sums = [], []
    for r in records:
        opencode_str = r.get('opencode')
        if opencode_str:
            try:
                nums = [int(x) for x in str(opencode_str).split(',')]
                tsum = sum(nums)
                groups.append(get_group_type(tsum))
                sums.append(tsum)
            except: continue
    return latest_issue, groups[::-1], sums[::-1]

def run_lightgbm_predictor(groups, sums):
    if len(sums) < 20: return None
    X, y = [], []
    for i in range(3, len(sums) - 1):
        feature = [sums[i], np.mean(sums[i-2:i+1]), sums[i-1] % 2, GROUP_MAP[groups[i]]]
        X.append(feature)
        y.append(GROUP_MAP[groups[i+1]])
    model = lgb.LGBMClassifier(n_estimators=30, learning_rate=0.05, max_depth=3, verbose=-1)
    model.fit(np.array(X), np.array(y))
    latest_feature = np.array([[sums[-1], np.mean(sums[-3:]), sums[-2] % 2, GROUP_MAP[groups[-1]]]])
    pred_idx = model.predict(latest_feature)[0]
    return REV_MAP[pred_idx]

def main_cloud_loop():
    last_issue = None
    while True:
        data = fetch_api_data(50)
        if data:
            issue, groups, sums = extract_history(data)
            if issue and issue != last_issue:
                last_issue = issue
                pred = run_lightgbm_predictor(groups, sums)
                msg = f"🔔 **期号: {issue}** | 开出: `{sums[-1]:02d}` ({groups[-1]})\n预测: {pred}"
                send_telegram_message(msg)
        time.sleep(10)

# ================= 极客自唤醒引擎 =================
app = Flask(__name__)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def keep_alive(path):
    return "🚀 量化穹顶引擎在线运行中...", 200

def run_flask_server():
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

def self_ping():
    while True:
        try:
            requests.get("https://quant-sniper.onrender.com", timeout=5)
        except: pass
        time.sleep(600) 

if __name__ == "__main__":
    send_telegram_message("✅ **云端要塞已部署并启动**")
    threading.Thread(target=run_flask_server, daemon=True).start()
    threading.Thread(target=self_ping, daemon=True).start()
    main_cloud_loop()

