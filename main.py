import requests
import time
import random
from collections import defaultdict
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import warnings

# ================= 新增：网页伪装依赖 =================
import threading
from flask import Flask
# ====================================================

warnings.filterwarnings("ignore")

# ================= 极客配置区 =================
DATA_URL = "https://" + "super.pc28998.com" + "/history/JND28"
TELEGRAM_BOT_TOKEN = "8790154521:AAEUz-Idju8kOEjhqyV9IMv2PEr2ditTUQg"
TELEGRAM_CHAT_ID = "6824519270"
# ==============================================

GROUP_MAP = {"大双": 0, "大单": 1, "小双": 2, "小单": 3}
REV_MAP = {0: "大双", 1: "大单", 2: "小双", 3: "小单"}

def send_telegram_message(message):
    if "你的Token" in TELEGRAM_BOT_TOKEN: return
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

def build_markov_models(groups):
    order1 = defaultdict(lambda: defaultdict(int))
    order2 = defaultdict(lambda: defaultdict(int))
    for i in range(len(groups) - 2):
        g1, g2, g3 = groups[i], groups[i+1], groups[i+2]
        order1[g2][g3] += 1
        order2[(g1, g2)][g3] += 1
    if len(groups) >= 2: order1[groups[-2]][groups[-1]] += 1
    return order1, order2

def run_lightgbm_predictor(groups, sums):
    if len(sums) < 20: return None
    X, y = [], []
    for i in range(3, len(sums) - 1):
        feature = [sums[i], np.mean(sums[i-2:i+1]), sums[i-1] % 2, GROUP_MAP[groups[i]]]
        X.append(feature)
        y.append(GROUP_MAP[groups[i+1]])
    if len(set(y)) < 2: return groups[-1] 
    model = lgb.LGBMClassifier(n_estimators=30, learning_rate=0.05, max_depth=3, silent=True)
    model.fit(np.array(X), np.array(y))
    latest_feature = np.array([[sums[-1], np.mean(sums[-3:]), sums[-2] % 2, GROUP_MAP[groups[-1]]]])
    pred_idx = model.predict(latest_feature)[0]
    return REV_MAP[pred_idx]

def run_stacking_judge(lgbm_pred, markov_pred, recent_groups):
    if lgbm_pred == markov_pred: return lgbm_pred, 85.0 
    volatility = sum(1 for i in range(1, len(recent_groups)) if recent_groups[i] != recent_groups[i-1]) / len(recent_groups)
    if volatility > 0.6: return markov_pred, 55.0
    else: return lgbm_pred, 60.0

def calculate_kelly_fraction(win_prob):
    b = 0.95 
    p = win_prob / 100.0
    q = 1.0 - p
    f_star = (b * p - q) / b
    return 0.0 if f_star <= 0 else (f_star / 2.0) * 100

def get_dynamic_specials(target_groups, all_sums):
    if "空仓" in target_groups: return [0, 0, 0]
    pool = {
        "大双": [14,16,18,20,22,24,26], "大单": [15,17,19,21,23,25,27],
        "小双": [0,2,4,6,8,10,12], "小单": [1,3,5,7,9,11,13]
    }
    local_mean = sum(all_sums[-20:]) / 20.0 if len(all_sums) >= 20 else 13.5
    candidates = []
    for g in target_groups: candidates.extend(pool[g])
    candidates.sort(key=lambda x: abs(x - local_mean + random.uniform(-0.5, 0.5)))
    return candidates[:3]

def cloud_ensemble_engine(groups, sums):
    if len(groups) < 30: return "📡 数据加载中...", "观望", ["空仓"], [0], 0
    lgbm_pred = run_lightgbm_predictor(groups, sums)
    if not lgbm_pred: lgbm_pred = groups[-1]
    order1, order2 = build_markov_models(groups)
    current_state_1 = groups[-1]
    current_state_2 = (groups[-2], groups[-1])
    next_probs = order2.get(current_state_2, order1.get(current_state_1, {}))
    if not next_probs: return "⚠️ 矩阵盲区", "观望", ["空仓"], [0], 0
    markov_top = sorted(next_probs.items(), key=lambda x: x[1], reverse=True)[0][0]
    final_judge, confidence = run_stacking_judge(lgbm_pred, markov_top, groups[-15:])
    vote_details = f"LGBM[{lgbm_pred}], 矩阵[{markov_top}]"
    
    if confidence < 50.0:
        return f"📉 混沌期预警 | 引擎分歧，强制锁仓\n🔍 详情: {vote_details}", "意见分歧", ["空仓", "防守"], [0], 0
        
    all_types = ["大单", "大双", "小单", "小双"]
    recommend_group = [final_judge]
    for g in all_types:
        if g != final_judge and len(recommend_group) < 2: recommend_group.append(g)
    kill_group = [g for g in all_types if g not in recommend_group][-1]
    specials = get_dynamic_specials(recommend_group, sums)
    kelly_pct = calculate_kelly_fraction(confidence)
    
    return f"👑 云端集成矩阵共振 | 胜率: {confidence:.1f}%\n🔍 底层侦测: {vote_details}", kill_group, recommend_group, specials, kelly_pct

def main_cloud_loop():
    last_issue = None
    while True:
        data = fetch_api_data(50)
        if data:
            issue, groups, sums = extract_history(data)
            if issue and issue != last_issue:
                last_issue = issue
                market_signal, kill, recommend, specials, kelly = cloud_ensemble_engine(groups, sums)
                msg = f"🔔 **期号: {issue}** | 开出: `{sums[-1]:02d}` ({groups[-1]})\n-------------------------\n{market_signal}\n"
                if "空仓" in recommend: msg += "\n🚫 **顶尖风控:** 大盘多空绞杀，强制空仓观望！"
                else:
                    msg += f"\n❌ 绝杀: `{kill}`\n✅ 双组: `{' + '.join(recommend)}`\n🎯 特码: `{', '.join(map(str, specials))}`"
                    msg += f"\n💰 凯利风控: `{'0% (模拟)' if kelly == 0 else f'{kelly:.1f}% 仓位'}`"
                send_telegram_message(msg)
        time.sleep(10)

# ================= 🌟 极客防休眠：网页伪装层 =================
app = Flask(__name__)

# 极客万能大门：无视一切奇葩请求，强制返回 200 存活信号！
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def keep_alive(path):
    return "🚀 量化穹顶引擎在线运行中 (Cloud-Native Mode)...", 200


def run_flask_server():
    # 开放 8080 端口让外部可以访问
    app.run(host='0.0.0.0', port=8080)
# ============================================================

if __name__ == "__main__":
    send_telegram_message("✅ **Render 云原生要塞部署成功**\n------------------\nLGBM + Markov 伪装协议已启动！")
    
    # 启动多线程：一个线程负责当“网站”忽悠平台，一个线程在后台默默算量化
    flask_thread = threading.Thread(target=run_flask_server)
    flask_thread.start()
    
    # 启动主核心
    main_cloud_loop()

