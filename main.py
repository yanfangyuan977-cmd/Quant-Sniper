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

POLL_INTERVAL = 60       # 每 60 秒轮询一次最新页面
MIN_DATA_REQUIRED = 300  
EXTREME_THRESHOLD = 0.08 
# =========================================================

client = pymongo.MongoClient(MONGO_URI)
db = client["pc28_quant_v3"]
collection = db["history_data"]

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
    """🛠️ 核心激光手术刀：清洗 HTML 网页源码并安全去重存入数据库"""
    pattern = r'<td[^>]*>(\d+)</td>\s*<td>\s*<span[^>]*>(\d+)</span>\s*<span[^>]*>(\d+)</span>\s*<span[^>]*>(\d+)</span>'
    records = re.findall(pattern, html_content)
    
    new_count = 0
    latest_issue = None
    
    # 逆序遍历，确保旧数据先处理，最新数据最后处理以获得准确的最新期号
    for item in reversed(records):
        issue = str(item[0])
        a, b, c = int(item[1]), int(item[2]), int(item[3])
        
        doc = {
            "_id": issue,
            "issue": issue,
            "A": a,
            "B": b,
            "C": c,
            "total": a + b + c,
            "timestamp": datetime.now()
        }
        
        result = collection.update_one({"_id": issue}, {"$setOnInsert": doc}, upsert=True)
        if result.upserted_id is not None:
            new_count += 1
            latest_issue = issue
            
    return new_count, latest_issue

def auto_backfill_5000_records():
    """🚀 V4.0 特种动作：启动时自动横扫 50 页历史时空，填满军火库"""
    print("⚡ 启动全自动历史时空回溯系统，正在合围前 50 页大盘底裤...", flush=True)
    send_telegram_msg("📡 **量化要塞启动中...**\n正在执行跨时空扫盘行动，全力吞噬 5000 期历史母体数据...")
    
    total_new_injected = 0
    for page in range(1, 51):
        url = f"https://www.jndpc.net/?ajax=1&tab=numbers&npage={page}"
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            data = res.json()
            html_content = data.get("html", "")
            
            new_added, _ = parse_and_save_html(html_content)
            total_new_injected += new_added
            
            if page % 10 == 0:
                print(f"⏳ 扫盘进度: {page}/50 页已吞噬...", flush=True)
            time.sleep(0.5)  # 云端优雅防封延时
        except Exception as e:
            print(f"⚠️ 扫盘在第 {page} 页遭遇微弱抵抗: {e}", flush=True)
            break
            
    final_total = collection.count_documents({})
    success_msg = f"🎉 **【时空回溯圆满大捷】** 🎉\n"
    success_msg += f"📥 本次新固化历史弹药: `{total_new_injected}` 期\n"
    success_msg += f"💎 云端要塞总储备现已达到: **{final_total}** 期！\n"
    success_msg += f"_算法大脑已吃饱和解构全部12天历史红利！_"
    send_telegram_msg(success_msg)

def build_micro_features(data_list):
    X_A, X_B, X_C = [], [], []
    y_A, y_B, y_C = [], [], []
    window = 5 
    for i in range(len(data_list) - window):
        hist = data_list[i:i+window]
        target = data_list[i+window]
        feat = []
        for h in hist:
            feat.extend([h['A'], h['B'], h['C']])
        X_A.append(feat)
        X_B.append(feat)
        X_C.append(feat)
        y_A.append(target['A'])
        y_B.append(target['B'])
        y_C.append(target['C'])
    return np.array(X_A), np.array(y_A), np.array(y_B), np.array(y_C)

def train_and_predict(data_list):
    X, y_A, y_B, y_C = build_micro_features(data_list)
    latest_feat = []
    for h in data_list[-5:]:
        latest_feat.extend([h['A'], h['B'], h['C']])
    latest_feat = np.array([latest_feat])
    
    params = {
        'objective': 'multiclass', 
        'num_class': 10, 
        'verbose': -1, 
        'seed': 42,
        'num_leaves': 15,
        'num_threads': 1  # 绝对锁定单线程，杜绝漂移
    }
    
    ds_A = lgb.Dataset(X, label=y_A)
    model_A = lgb.train(params, ds_A, num_boost_round=30)
    prob_A = model_A.predict(latest_feat)[0]
    
    ds_B = lgb.Dataset(X, label=y_B)
    model_B = lgb.train(params, ds_B, num_boost_round=30)
    prob_B = model_B.predict(latest_feat)[0]
    
    ds_C = lgb.Dataset(X, label=y_C)
    model_C = lgb.train(params, ds_C, num_boost_round=30)
    prob_C = model_C.predict(latest_feat)[0]
    
    return prob_A, prob_B, prob_C

def get_attr(num):
    size = "大" if num >= 5 else "小"
    parity = "单" if num % 2 != 0 else "双"
    return f"{size}{parity}"

def run_quant_engine():
    print("🚀 V4.0 全自动时空回溯要塞点火...", flush=True)
    
    # ⚡ 步骤一：启动即执行 50 页全量历史吞噬
    auto_backfill_5000_records()
    
    send_telegram_msg("🟢 **V4.0 终极要塞完全体已上线**\n【全量实时同步雷达】开始进入每 60 秒不间断巡航状态！")
    last_issue_alerted = None
    
    while True:
        try:
            # ⚡ 步骤二：每 60 秒高频锁定最新第 1 页抓取
            live_url = "https://www.jndpc.net/?ajax=1&tab=numbers&npage=1"
            res = requests.get(live_url, headers=HEADERS, timeout=10)
            data = res.json()
            html_content = data.get("html", "")
            
            new_added, latest_issue = parse_and_save_html(html_content)
            total_count = collection.count_documents({})
            
            # 如果没抓出最新期号，从数据库兜底查最新一条
            if not latest_issue:
                last_doc = collection.find_one(sort=[("_id", pymongo.DESCENDING)])
                if last_doc:
                    latest_issue = last_doc["issue"]
            
            # 只有当新抓取到数据，且是一个全新未预测过的期号时，触发开火
            if latest_issue and latest_issue != last_issue_alerted:
                print(f"🎯 捕获新实盘轨迹期号: {latest_issue} | 当前数据库储备: {total_count}期", flush=True)
                
                cursor = collection.find().sort("_id", 1)
                data_list = list(cursor)
                
                # 算法起算
                prob_A, prob_B, prob_C = train_and_predict(data_list)
                
                pred_A = int(np.argmax(prob_A))
                pred_B = int(np.argmax(prob_B))
                pred_C = int(np.argmax(prob_C))
                
                total_probs = np.zeros(28)
                for i in range(10):
                    for j in range(10):
                        for k in range(10):
                            total_probs[i+j+k] += prob_A[i] * prob_B[j] * prob_C[k]
                            
                prob_small = sum(total_probs[0:14])
                prob_big = sum(total_probs[14:28])
                prob_odd = sum(total_probs[i] for i in range(28) if i % 2 != 0)
                prob_even = sum(total_probs[i] for i in range(28) if i % 2 == 0)
                prob_extreme = sum(total_probs[0:6]) + sum(total_probs[22:28])
                
                next_issue = str(int(latest_issue) + 1)
                msg = f"🔔 期号: `{next_issue}` | 预测战报\n"
                msg += "-" * 25 + "\n"
                msg += f"🔍 样本池: `{total_count}`期 (黄金源)\n\n"
                msg += "🎯 **【ABC 微观狙击】**\n"
                msg += f"A区: `{pred_A}` ({get_attr(pred_A)}) | 胜率: {max(prob_A)*100:.1f}%\n"
                msg += f"B区: `{pred_B}` ({get_attr(pred_B)}) | 胜率: {max(prob_B)*100:.1f}%\n"
                msg += f"C区: `{pred_C}` ({get_attr(pred_C)}) | 胜率: {max(prob_C)*100:.1f}%\n\n"
                
                msg += "🎲 **【宏观概率合围】**\n"
                best_combo = ""
                if prob_big > prob_small and prob_even > prob_odd: best_combo = "大双"
                elif prob_big > prob_small and prob_odd > prob_even: best_combo = "大单"
                elif prob_small > prob_big and prob_even > prob_odd: best_combo = "小双"
                else: best_combo = "小单"
                
                msg += f"✅ 核心双组: **{best_combo}**\n"
                
                if prob_extreme >= EXTREME_THRESHOLD:
                    msg += f"\n🚨 **【深海鱼雷警报】** 🚨\n"
                    msg += f"极值爆发概率异常: **{prob_extreme*100:.1f}%**\n"
                    msg += "建议：小仓防守极大/极小！\n"
                    
                send_telegram_msg(msg)
                last_issue_alerted = latest_issue
                
        except Exception as e:
            print(f"⚠️ 核心循环发生短暂网络抖动: {e}", flush=True)
            
        time.sleep(POLL_INTERVAL)

app = Flask(__name__)

@app.route("/")
def keep_alive():
    return "🚀 V4.0 时空回溯要塞最高完全体已就位...", 200

def run_flask_server():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, use_reloader=False)

if __name__ == "__main__":
    threading.Thread(target=run_flask_server, daemon=True).start()
    run_quant_engine()

