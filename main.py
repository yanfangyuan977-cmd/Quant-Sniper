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
# 💡 漏洞修复：启用全新精纯集合，避免与旧文本格式的 _id 发生冲突
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
        # 💡 漏洞修复：将期号强转为 int 型，彻底根除字符串字典序排序隐患
        issue_int = int(item[0])
        a, b, c = int(item[1]), int(item[2]), int(item[3])
        
        doc = {
            "_id": issue_int,     # 固化为数字主键
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
    print("⚡ 启动全自动历史时空回溯系统，正在合围前 50 页大盘底裤...", flush=True)
    send_telegram_msg("📡 **量化要塞启动中...**\n正在执行跨时空扫盘行动，全力吞噬历史母体数据...")
    
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
            time.sleep(0.5)
        except Exception as e:
            print(f"⚠️ 扫盘在第 {page} 页遭遇微弱抵抗: {e}", flush=True)
            break
            
    final_total = collection.count_documents({})
    success_msg = f"🎉 **【时空回溯防弹版大捷】** 🎉\n"
    success_msg += f"📥 精纯数字弹药成功固化: `{total_new_injected}` 期\n"
    success_msg += f"💎 纯整型索引要塞总储备: **{final_total}** 期！\n"
    success_msg += f"_彻底根除排序漏洞，算法大脑进入无瑕模式！_"
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
        'num_threads': 1  
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
    print("🚀 V4.1 纯净整型对冲要塞启动...", flush=True)
    auto_backfill_5000_records()
    
    send_telegram_msg("🟢 **V4.1 终极无瑕要塞已上线**\n【数字整型排序防御网】已全面固化！")
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
                # 💡 漏洞修复：数字整型完美无误差进行降序兜底查询
                last_doc = collection.find_one(sort=[("_id", pymongo.DESCENDING)])
                if last_doc:
                    latest_issue = str(last_doc["_id"])
            
            if latest_issue and latest_issue != last_issue_alerted:
                print(f"🎯 捕获新期号: {latest_issue} | 数据库储备: {total_count}期", flush=True)
                
                # 💡 漏洞修复：数字主键原生升序排列，速度极快，顺序100%精准
                cursor = collection.find().sort("_id", 1)
                data_list = list(cursor)
                
                prob_A, prob_B, prob_C = train_and_predict(data_list)
                
                pred_A = int(np.argmax(prob_A))
                pred_B = int(np.argmax(prob_B))
                pred_C = int(np.argmax(prob_C))
                
                total_probs = np.zeros(28)
                for i in range(10):
                    for j in range(10):
                        for k in range(10):
                            total_probs[i+j+k] += prob_A[i] * prob_B[j] * prob_C[k]
                            
                p_big_even = float(sum(total_probs[m] for m in range(14, 28) if m % 2 == 0))   
                p_big_odd = float(sum(total_probs[m] for m in range(14, 28) if m % 2 != 0))    
                p_small_even = float(sum(total_probs[m] for m in range(0, 14) if m % 2 == 0))  
                p_small_odd = float(sum(total_probs[m] for m in range(0, 14) if m % 2 != 0))   
                
                combos = [
                    ("大双", p_big_even),
                    ("大单", p_big_odd),
                    ("小双", p_small_even),
                    ("小单", p_small_odd)
                ]
                combos.sort(key=lambda x: x[1], reverse=True)  
                
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
                msg += f"🥇 核心首选: **{combos[0][0]}** | 胜率: {combos[0][1]*100:.1f}%\n"
                msg += f"🥈 战术对冲: **{combos[1][0]}** | 胜率: {combos[1][1]*100:.1f}%\n\n"
                
                if prob_extreme >= EXTREME_THRESHOLD:
                    msg += f"🚨 **【深海鱼雷警报】** 🚨\n"
                    msg += f"极值爆发概率异常: **{prob_extreme*100:.1f}%**\n"
                    msg += "建议：小仓防守极大/极小！\n"
                    
                send_telegram_msg(msg)
                last_issue_alerted = latest_issue
                
        except Exception as e:
            print(f"⚠️ 核心循环网络抖动: {e}", flush=True)
            
        time.sleep(POLL_INTERVAL)

app = Flask(__name__)

@app.route("/")
def keep_alive():
    return "🚀 V4.1 终极排雷防御要塞正在稳定巡航...", 200

def run_flask_server():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, use_reloader=False)

if __name__ == "__main__":
    threading.Thread(target=run_flask_server, daemon=True).start()
    run_quant_engine()

