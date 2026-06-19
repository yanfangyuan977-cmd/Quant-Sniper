import time
import requests
import pymongo
import numpy as np
import lightgbm as lgb
from datetime import datetime
import os
import threading
import re
from flask import Flask

# ================= 🎛️ 终极量化控制台 🎛️ =================
BOT_TOKEN = "8790154521:AAEUz-Idju8kOEjhqyV9IMv2PEr2ditTUQg"
CHAT_ID = "6824519270"

# 固化云端 MongoDB 通道
MONGO_URI = "mongodb+srv://admin:xiaoxin520@cluster0.apmxxbi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

POLL_INTERVAL = 60       
MIN_DATA_REQUIRED = 300  
# =========================================================

client = pymongo.MongoClient(MONGO_URI)
db = client["pc28_quant_v3"]
collection = db["history_data_v4"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "X-Requested-With": "XMLHttpRequest"
}

def send_telegram_msg(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=5)
    except:
        pass

def parse_and_save_html(html_content):
    pattern = r'<td[^>]*>(\d+)</td>\s*<td>\s*<span[^>]*>(\d+)</span>\s*<span[^>]*>(\d+)</span>\s*<span[^>]*>(\d+)</span>'
    records = re.findall(pattern, html_content)
    
    new_count = 0
    latest_issue = None
    
    for item in reversed(records):
        issue_int = int(item[0])
        a, b, c = int(item[1]), int(item[2]), int(item[3])
        
        doc = {
            "_id": issue_int,     
            "issue": str(issue_int),
            "A": a,
            "B": b,
            "C": c,
            "total": a + b + c,
            "timestamp": datetime.now()
        }
        
        result = collection.update_one({"_id": issue_int}, {"$setOnInsert": doc}, upsert=True)
        if result.upserted_id is not None:
            new_count += 1
            latest_issue = str(issue_int)
            
    return new_count, latest_issue

def auto_backfill_5000_records():
    print("⚡ 启动全自动历史时空回溯系统...", flush=True)
    send_telegram_msg("📡 **V5.0 天王星宏观直分类版部署中...**\n核心重炮正在进行底层数据链深度咬合...")
    
    total_new_injected = 0
    for page in range(1, 51):
        url = f"https://www.jndpc.net/?ajax=1&tab=numbers&npage={page}"
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            data = res.json()
            html_content = data.get("html", "")
            
            new_added, _ = parse_and_save_html(html_content)
            total_new_injected += new_added
            time.sleep(0.5)
        except Exception as e:
            print(f"⚠️ 扫盘在第 {page} 页遭遇微弱抵抗: {e}", flush=True)
            break
            
    final_total = collection.count_documents({})
    success_msg = f"🎉 **【V5.0 宏观直分类版要塞合龙】** 🎉\n"
    success_msg += f"📥 固化历史母体总规模: **{final_total}** 期！\n"
    success_msg += f"_💡 彻底废除ABC微观连乘错误，双重宏观大局观雷达已锁死目标！_"
    send_telegram_msg(success_msg)

def extract_features_from_slice(hist_slice):
    """
    💡 核心解耦复核：提取确定维度的特征向量。
    包含：最后5期的微观特征，以及整段历史（10期）的宏观滚动气候。
    """
    feat = []
    # 1. 抽取最后5期的微观物理状态
    for h in hist_slice[-5:]:
        span = max(h['A'], h['B'], h['C']) - min(h['A'], h['B'], h['C'])
        feat.extend([h['A'], h['B'], h['C'], h['total'], span]) # 5期 * 5 = 25个特征
        
    # 2. 注入方案二最强的大局观大盘宏观气候（基于整段10期）
    totals = [x['total'] for x in hist_slice]
    rolling_mean = float(np.mean(totals))
    rolling_std = float(np.std(totals))
    feat.extend([rolling_mean, rolling_std]) # 2个特征
    
    return feat # 固定维度：25 + 2 = 27维特征，严防任何反向提升

def build_macro_dataset(data_list):
    X = []
    y_combo = []
    y_special = []
    window = 10 # 滚动时间窗口拉长到10，提供大局观背景
    
    for i in range(len(data_list) - window):
        hist_slice = data_list[i : i + window]
        target_item = data_list[i + window]
        
        # 提取27维特征
        feat = extract_features_from_slice(hist_slice)
        X.append(feat)
        
        # 彻底砍掉中间商，直接对最终宏观目标进行硬映射
        total = target_item['total']
        is_big = 1 if total >= 14 else 0
        is_odd = 1 if total % 2 != 0 else 0
        
        # 组合映射编码：0:大单, 1:大双, 2:小单, 3:小双
        if is_big and is_odd: combo_code = 0
        elif is_big and not is_odd: combo_code = 1
        elif not is_big and is_odd: combo_code = 2
        else: combo_code = 3
        
        y_combo.append(combo_code)
        y_special.append(total) # 特码直接映射 0 - 27 分类
        
    return np.array(X), np.array(y_combo), np.array(y_special)

def train_and_predict_macro(data_list):
    X, y_combo, y_special = build_macro_dataset(data_list)
    
    # 提取当前最新的27维特征
    latest_feat = np.array([extract_features_from_slice(data_list[-10:])])
    
    # 💡 军工级参数防线，配合1717期大数据，强力卡死任何过拟合噪音
    base_params = {
        'verbose': -1, 
        'seed': 42,
        'num_leaves': 31,
        'num_threads': 1,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 35
    }
    
    # 🔥 重炮一：组合4分类直接预测器
    params_combo = base_params.copy()
    params_combo.update({'objective': 'multiclass', 'num_class': 4})
    ds_combo = lgb.Dataset(X, label=y_combo)
    model_combo = lgb.train(params_combo, ds_combo, num_boost_round=35)
    prob_combo = model_combo.predict(latest_feat)[0] # 长度为4的宏观纯净概率数组
    
    # 🔥 重炮二：特码28分类直接预测器
    params_special = base_params.copy()
    params_special.update({'objective': 'multiclass', 'num_class': 28})
    ds_special = lgb.Dataset(X, label=y_special)
    model_special = lgb.train(params_special, ds_special, num_boost_round=35)
    prob_special = model_special.predict(latest_feat)[0] # 长度为28的特码纯净概率数组
    
    return prob_combo, prob_special

def run_quant_engine():
    print("🚀 V5.0 天王星宏观直分类狙击要塞点火...", flush=True)
    auto_backfill_5000_records()
    
    send_telegram_msg("🟢 **V5.0 宏观直分类主炮全面校准**\n【组合4分类 & 特码28分类级独立制导系统】进入不间断战备！")
    last_issue_alerted = None
    
    while True:
        try:
            live_url = "https://www.jndpc.net/?ajax=1&tab=numbers&npage=1"
            res = requests.get(live_url, headers=HEADERS, timeout=10)
            data = res.json()
            html_content = data.get("html", "")
            
            new_added, latest_issue = parse_and_save_html(html_content)
            total_count = collection.count_documents({})
            
            if not latest_issue:
                last_doc = collection.find_one(sort=[("_id", pymongo.DESCENDING)])
                if last_doc:
                    latest_issue = str(last_doc["_id"])
            
            if latest_issue and latest_issue != last_issue_alerted:
                print(f"🎯 捕获新期号: {latest_issue} | 数据库储备: {total_count}期", flush=True)
                
                cursor = collection.find().sort("_id", 1)
                data_list = list(cursor)
                
                # 执行V5.0终极宏观直分类预测
                prob_combo, prob_special = train_and_predict_macro(data_list)
                
                # 解析预测组合排序
                combo_names = ["大单", "大双", "小单", "小双"]
                combos = [(combo_names[i], float(prob_combo[i])) for i in range(4)]
                combos.sort(key=lambda x: x[1], reverse=True) # 纯天然无连乘损耗的宏观胜率
                
                # 解析特码绝对前三名
                top_3_special_idx = np.argsort(prob_special)[-3:][::-1]
                
                next_issue = str(int(latest_issue) + 1)
                msg = f"🔔 期号: `{next_issue}` | V5.0 天王星宏观战报\n"
                msg += "-" * 25 + "\n"
                msg += f"🔍 战略数据源: `{total_count}`期 (大样本蓄能)\n\n"
                
                msg += "🎲 **【🔥 宏观直接分类制导】**\n"
                msg += f"🥇 核心首选: **{combos[0][0]}** | 纯净胜率: {combos[0][1]*100:.1f}%\n"
                msg += f"🥈 战术次选: **{combos[1][0]}** | 纯净胜率: {combos[1][1]*100:.1f}%\n\n"
                
                msg += "🎯 **【🔥 特码直接分类狙击】**\n"
                msg += f"🥇 狙击一号: **{top_3_special_idx[0]}点** | 独立胜率: {prob_special[top_3_special_idx[0]]*100:.1f}%\n"
                msg += f"🥈 狙击二号: **{top_3_special_idx[1]}点** | 独立胜率: {prob_special[top_3_special_idx[1]]*100:.1f}%\n"
                msg += f"🥉 狙击三号: **{top_3_special_idx[2]}点** | 独立胜率: {prob_special[top_3_special_idx[2]]*100:.1f}%\n"
                
                send_telegram_msg(msg)
                last_issue_alerted = latest_issue
                
        except Exception as e:
            print(f"⚠️ 核心循环网络抖动: {e}", flush=True)
            
        time.sleep(POLL_INTERVAL)

app = Flask(__name__)

@app.route("/")
def keep_alive():
    return "🚀 V5.0 天王星宏观直分类完全体正在最高效航行...", 200

def run_flask_server():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, use_reloader=False)

if __name__ == "__main__":
    threading.Thread(target=run_flask_server, daemon=True).start()
    run_quant_engine()

