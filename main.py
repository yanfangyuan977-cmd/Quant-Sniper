import time
import requests
import pymongo
import numpy as np
import lightgbm as lgb
from datetime import datetime
import os
import threading
from flask import Flask

# ================= 🎛️ 终极战术控制台 🎛️ =================
# ⚠️ 部署前必须替换为你自己的真实配置！
BOT_TOKEN = "8790154521:AAEUz-Idju8kOEjhqyV9IMv2PEr2ditTUQg"
CHAT_ID = "6824519270"
DATA_URL = "https://super.pc28998.com/history/JND28?limit=60"

# MongoDB 云端武器库秘钥 (已修正为小写 m)
MONGO_URI = "mongodb+srv://admin:xiaoxin520@cluster0.apmxxbi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# 核心战术参数
POLL_INTERVAL = 20       # 每 20 秒拉取一次数据防漏
MIN_DATA_REQUIRED = 300  # 方案B：蓄水池最低启动期数
EXTREME_THRESHOLD = 0.08 # 极值警报触发阈值 (爆发概率 > 8% 触发深海鱼雷)
# =========================================================

# 连接云端武器库
client = pymongo.MongoClient(MONGO_URI)
db = client["pc28_quant_v3"]
collection = db["history_data"]

def send_telegram_msg(text):
    """发送战报到 Telegram"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=5)
    except:
        pass

def fetch_and_store_data():
    """极速侦测并写入云端，返回最新一期的期号和特征库数量"""
    try:
        res = requests.get(DATA_URL, timeout=5)
        data = res.json()
        
        new_count = 0
        latest_issue = None
        
        # 假设返回的是列表，倒序遍历保证按时间顺序插入
        for item in reversed(data):
            issue = str(item.get("issue", ""))
            opencode = str(item.get("opencode", ""))
            if not issue or not opencode:
                continue
                
            nums = [int(x) for x in opencode.split(",")]
            if len(nums) != 3:
                continue
                
            doc = {
                "_id": issue, 
                "issue": issue,
                "A": nums[0],
                "B": nums[1],
                "C": nums[2],
                "total": sum(nums),
                "timestamp": datetime.now()
            }
            
            result = collection.update_one({"_id": issue}, {"$setOnInsert": doc}, upsert=True)
            if result.upserted_id is not None:
                new_count += 1
                latest_issue = issue
                
        total_count = collection.count_documents({})
        return new_count, total_count, latest_issue
    except Exception as e:
        print(f"侦测失败: {e}")
        return 0, collection.count_documents({}), None

def build_micro_features(data_list):
    """ABC 三区微观独立特征工程"""
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
    """路线 Y：训练三大独立模型，输出概率矩阵"""
    X, y_A, y_B, y_C = build_micro_features(data_list)
    
    latest_feat = []
    for h in data_list[-5:]:
        latest_feat.extend([h['A'], h['B'], h['C']])
    latest_feat = np.array([latest_feat])
    
    params = {'objective': 'multiclass', 'num_class': 10, 'verbose': -1, 'seed': 42}
    
    ds_A = lgb.Dataset(X, label=y_A)
    model_A = lgb.train(params, ds_A, num_boost_round=50)
    prob_A = model_A.predict(latest_feat)[0]
    
    ds_B = lgb.Dataset(X, label=y_B)
    model_B = lgb.train(params, ds_B, num_boost_round=50)
    prob_B = model_B.predict(latest_feat)[0]
    
    ds_C = lgb.Dataset(X, label=y_C)
    model_C = lgb.train(params, ds_C, num_boost_round=50)
    prob_C = model_C.predict(latest_feat)[0]
    
    return prob_A, prob_B, prob_C

def get_attr(num):
    size = "大" if num >= 5 else "小"
    parity = "单" if num % 2 != 0 else "双"
    return f"{size}{parity}"

def run_quant_engine():
    """主侦测循环引擎"""
    print("🚀 V3.1 微观量化要塞启动...")
    send_telegram_msg("🟢 **V3.1 终极要塞已上线**\n云端数据库连接成功，雷达启动！")
    
    last_issue_alerted = None
    
    while True:
        new_added, total_count, latest_issue = fetch_and_store_data()
        
        if new_added == 0:
            time.sleep(POLL_INTERVAL)
            continue
            
        print(f"侦测到新期号: {latest_issue} | 云端总弹药: {total_count}期")
        
        # ====== 方案 B：苦行僧静默锁 ======
        if total_count < MIN_DATA_REQUIRED:
            if latest_issue != last_issue_alerted:
                if total_count % 10 == 0:
                    send_telegram_msg(f"🔋 **武器充能中...**\n当前弹药: `{total_count} / {MIN_DATA_REQUIRED}` 期\n_方案B强制静默收集，不提供预测以保护资金_")
                last_issue_alerted = latest_issue
            time.sleep(POLL_INTERVAL)
            continue
            
        # ====== 充能完毕，全功率开火 ======
        if latest_issue != last_issue_alerted:
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
                        
            prob_small = sum(total_probs[0:14])
            prob_big = sum(total_probs[14:28])
            prob_odd = sum(total_probs[i] for i in range(28) if i % 2 != 0)
            prob_even = sum(total_probs[i] for i in range(28) if i % 2 == 0)
            prob_extreme = sum(total_probs[0:6]) + sum(total_probs[22:28])
            
            next_issue = str(int(latest_issue) + 1)
            msg = f"🔔 期号: `{next_issue}` | 预测预测\n"
            msg += "-" * 25 + "\n"
            msg += f"🔍 样本池: `{total_count}`期 (云端永固)\n\n"
            
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
                msg += f"极值(0-5, 22-27) 爆发概率异常: **{prob_extreme*100:.1f}%**\n"
                msg += "建议：小仓位防守极大/极小，博取 17.5 倍赔率！\n"
                
            send_telegram_msg(msg)
            last_issue_alerted = latest_issue
            
        time.sleep(POLL_INTERVAL)

# =========================================================
# 🛡️ Render 防崩溃伪装装甲 (Web Server)
# =========================================================
app = Flask(__name__)

@app.route("/")
def keep_alive():
    return "🚀 V3.1 微观量化要塞 (云端不灭版) 正常运行中...", 200

def run_flask_server():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, use_reloader=False)

if __name__ == "__main__":
    # 1. 开启伪装服务应付 Render 端口检测
    threading.Thread(target=run_flask_server, daemon=True).start()
    
    # 2. 启动核心量化引擎
    run_quant_engine()

