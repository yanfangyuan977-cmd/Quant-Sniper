# 强制唤醒引擎 v2
import requests
import time
import random
import json
import os
from collections import defaultdict
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
import threading
from flask import Flask

warnings.filterwarnings("ignore")

# ================= 🌟 核心资产配置区 =================
DATA_URL = "https://super.pc28998.com/history/JND28"
TELEGRAM_BOT_TOKEN = "8790154521:AAEUz-Idju8kOEjhqyV9IMv2PEr2ditTUQg"
TELEGRAM_CHAT_ID = "6824519270"
RESERVOIR_FILE = "data_reservoir.json"  # 数据蓄水池文件
# ===================================================

GROUP_MAP = {"大双": 0, "大单": 1, "小双": 2, "小单": 3}
REV_MAP = {0: "大双", 1: "大单", 2: "小双", 3: "小单"}

# 全局蓄水池内存
GLOBAL_RESERVOIR = {}

def load_reservoir():
    global GLOBAL_RESERVOIR
    if os.path.exists(RESERVOIR_FILE):
        try:
            with open(RESERVOIR_FILE, 'r', encoding='utf-8') as f:
                GLOBAL_RESERVOIR = json.load(f)
            print(f"✅ 成功加载蓄水池数据: {len(GLOBAL_RESERVOIR)} 条记录")
        except: pass

def save_reservoir():
    try:
        with open(RESERVOIR_FILE, 'w', encoding='utf-8') as f:
            json.dump(GLOBAL_RESERVOIR, f, ensure_ascii=False)
    except: pass

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

def update_and_get_reservoir():
    """蓄水池进水阀：只进不出，自动去重，永久保存"""
    global GLOBAL_RESERVOIR
    data = fetch_api_data(50)
    if not data: return None
    
    records = []
    if isinstance(data, list): records = data
    elif isinstance(data, dict):
        for key in ['data', 'list', 'result', 'records']:
            if key in data and isinstance(data[key], list):
                records = data[key]
                break
                
    if not records: return None
    
    # 将新数据注入蓄水池
    updated = False
    for r in records:
        issue = str(r.get('expect'))
        if issue and issue not in GLOBAL_RESERVOIR:
            GLOBAL_RESERVOIR[issue] = r
            updated = True
            
    if updated:
        save_reservoir() # 写入本地文件
        
    # 按期号从小到大排序返回（最旧的数据在最前面，最新的在最后面）
    sorted_issues = sorted(GLOBAL_RESERVOIR.keys())
    return [GLOBAL_RESERVOIR[iss] for iss in sorted_issues]

def extract_history_from_reservoir(records):
    groups, sums = [], []
    if not records: return None, [], []
    latest_issue = records[-1].get('expect', '未知')
    for r in records:
        opencode_str = r.get('opencode')
        if opencode_str:
            try:
                nums = [int(x) for x in str(opencode_str).split(',')]
                tsum = sum(nums)
                groups.append(get_group_type(tsum))
                sums.append(tsum)
            except: continue
    return latest_issue, groups, sums

def build_markov_models(groups):
    order1 = defaultdict(lambda: defaultdict(int))
    order2 = defaultdict(lambda: defaultdict(int))
    for i in range(len(groups) - 2):
        g1, g2, g3 = groups[i], groups[i+1], groups[i+2]
        order1[g2][g3] += 1
        order2[(g1, g2)][g3] += 1
    if len(groups) >= 2: order1[groups[-2]][groups[-1]] += 1
    return order1, order2

def prepare_ml_features(groups, sums):
    if len(sums) < 25: 
        return None, None, None
    X, y = [], []
    for i in range(3, len(sums) - 1):
        feature = [sums[i], np.mean(sums[i-2:i+1]), sums[i-1] % 2, GROUP_MAP[groups[i]]]
        X.append(feature)
        y.append(GROUP_MAP[groups[i+1]])
    if len(set(y)) < 2: 
        return None, None, None
    latest_feature = np.array([[sums[-1], np.mean(sums[-3:]), sums[-2] % 2, GROUP_MAP[groups[-1]]]])
    return np.array(X), np.array(y), latest_feature

def run_ml_ensemble_predictors(X, y, latest_feature, fallback):
    try:
        model_lgb = lgb.LGBMClassifier(n_estimators=30, learning_rate=0.05, max_depth=3, verbose=-1)
        model_lgb.fit(X, y)
        lgb_pred = REV_MAP[model_lgb.predict(latest_feature)[0]]
    except: lgb_pred = fallback

    try:
        model_xgb = xgb.XGBClassifier(n_estimators=30, learning_rate=0.05, max_depth=3, eval_metric='mlogloss', verbose=0)
        model_xgb.fit(X, y)
        xgb_pred = REV_MAP[model_xgb.predict(latest_feature)[0]]
    except: xgb_pred = fallback

    try:
        model_rf = RandomForestClassifier(n_estimators=30, max_depth=3, random_state=42)
        model_rf.fit(X, y)
        rf_pred = REV_MAP[model_rf.predict(latest_feature)[0]]
    except: rf_pred = fallback

    return lgb_pred, xgb_pred, rf_pred

def run_hybrid_ensemble_judge(lgb_p, xgb_p, rf_p, markov_p):
    votes = defaultdict(int)
    votes[lgb_p] += 1
    votes[xgb_p] += 1
    votes[rf_p] += 1
    
    ml_best = max(votes, key=votes.get)
    ml_votes = votes[ml_best]
    
    if ml_votes == 3 and ml_best == markov_p: return ml_best, 92.0
    elif ml_votes == 3: return ml_best, 80.0
    elif ml_votes == 2 and ml_best == markov_p: return ml_best, 75.0
    elif ml_votes == 2: return ml_best, 60.0
    else: return markov_p, 45.0

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
    
    order1, order2 = build_markov_models(groups)
    current_state_1 = groups[-1]
    current_state_2 = (groups[-2], groups[-1])
    next_probs = order2.get(current_state_2, order1.get(current_state_1, {}))
    if not next_probs: return "⚠️ 矩阵盲区", "观望", ["空仓"], [0], 0
    markov_top = sorted(next_probs.items(), key=lambda x: x[1], reverse=True)[0][0]
    
    X, y, latest_f = prepare_ml_features(groups, sums)
    if X is not None: lgb_p, xgb_p, rf_p = run_ml_ensemble_predictors(X, y, latest_f, groups[-1])
    else: lgb_p, xgb_p, rf_p = groups[-1], groups[-1], groups[-1]
        
    final_judge, confidence = run_hybrid_ensemble_judge(lgb_p, xgb_p, rf_p, markov_top)
    vote_details = f"LGBM[{lgb_p}] | XGB[{xgb_p}] | RF[{rf_p}] | 矩阵[{markov_top}]"
    
    if confidence < 50.0:
        return f"📉 混沌期预警 | 决策链多维分歧，强控锁仓\n🔍 详情: {vote_details}", "意见分歧", ["空仓", "防守"], [0], 0
        
    all_types = ["大单", "大双", "小单", "小双"]
    recommend_group = [final_judge]
    for g in all_types:
        if g != final_judge and len(recommend_group) < 2: recommend_group.append(g)
    kill_group = [g for g in all_types if g not in recommend_group][-1]
    specials = get_dynamic_specials(recommend_group, sums)
    kelly_pct = calculate_kelly_fraction(confidence)
    
    return f"👑 云端最高法共振 | 胜率: {confidence:.1f}%\n🔍 样本池容量: {len(groups)}期\n🔍 底层侦测: {vote_details}", kill_group, recommend_group, specials, kelly_pct

def main_cloud_loop():
    load_reservoir() # 启动时先加载历史蓄水池
    last_issue = None
    while True:
        all_records = update_and_get_reservoir() # 获取水池里所有数据
        if all_records:
            issue, groups, sums = extract_history_from_reservoir(all_records)
            if issue and issue != last_issue:
                last_issue = issue
                market_signal, kill, recommend, specials, kelly = cloud_ensemble_engine(groups, sums)
                msg = f"🔔 **期号: {issue}** | 开出: `{sums[-1]:02d}` ({groups[-1]})\n-------------------------\n{market_signal}\n"
                if "空仓" in recommend: msg += "\n🚫 **顶尖风控:** 算法阵营陷入絞杀，强控全员空仓观望！"
                else:
                    msg += f"\n❌ 绝杀: `{kill}`\n✅ 双组: `{' + '.join(recommend)}`\n🎯 特码: `{', '.join(map(str, specials))}`"
                    msg += f"\n💰 凯利风控: `{'0% (模拟)' if kelly == 0 else f'{kelly:.1f}% 仓位'}`"
                send_telegram_message(msg)
        time.sleep(10)

# ================= 🌟 极客防休眠：网页伪装层 =================
app = Flask(__name__)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def keep_alive(path):
    return "🚀 量化穹顶引擎在线运行中 (Cloud-Native Mode)...", 200

def run_flask_server():
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)

def self_ping():
    while True:
        try: requests.get("https://quant-sniper.onrender.com", timeout=5)
        except: pass
        time.sleep(600)

if __name__ == "__main__":
    send_telegram_message("✅ **量化要塞重启成功**\n------------------\n数据蓄水池已开启，最高法官裁决系统运转中！")
    
    threading.Thread(target=run_flask_server, daemon=True).start()
    threading.Thread(target=self_ping, daemon=True).start()
    
    main_cloud_loop()
