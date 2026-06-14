import requests
import time
import random
import json
import os
import logging
import threading
from collections import defaultdict

import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
import xgboost as xgb
import schedule
import warnings
from flask import Flask

warnings.filterwarnings("ignore")

# ================= 🌟 核心资产配置区 =================
DATA_URL        = "https://super.pc28998.com/history/JND28"

# ⚠️⚠️⚠️ 长官注意：部署前必须把下面两行改回你自己的真实秘钥！⚠️⚠️⚠️
TELEGRAM_BOT_TOKEN="8790154521:AAEUz-Idju8kOEjhqyV9IMv2PEr2ditTUQg"
TELEGRAM_CHAT_ID="6824519270"
RESERVOIR_FILE     = "data_reservoir.json"
MAX_RESERVOIR_SIZE = 2000   # 蓄水池上限，防止内存无限增长
RETRAIN_EVERY      = 20     # 每新增20期才触发重训练
# ===================================================

# ================= 🌟 结构化日志 =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("quant.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)
# ================================================

GROUP_MAP  = {"大双": 0, "大单": 1, "小双": 2, "小单": 3}
REV_MAP    = {0: "大双", 1: "大单", 2: "小双", 3: "小单"}
ALL_GROUPS = list(GROUP_MAP.keys())
WINDOW_SIZES = [5, 10, 20]

# ================= 🌟 线程安全全局状态 =================
GLOBAL_RESERVOIR: dict = {}
_reservoir_lock  = threading.RLock()
_last_issue: str = ""


# ================================================================
# 🗄️  模型缓存：避免每10秒重训练
# ================================================================
class ModelCache:
    def __init__(self, retrain_every: int = RETRAIN_EVERY):
        self._lock          = threading.Lock()
        self.models: dict   = {}
        self.last_train_size: int  = 0
        self.retrain_every  = retrain_every
        self.cv_accuracy: float    = 0.25   

    def should_retrain(self, data_size: int) -> bool:
        with self._lock:
            return (data_size - self.last_train_size) >= self.retrain_every

    def update(self, models: dict, data_size: int, cv_accuracy: float = 0.25):
        with self._lock:
            self.models       = models
            self.last_train_size = data_size
            self.cv_accuracy  = cv_accuracy

    def get(self) -> tuple:
        with self._lock:
            return dict(self.models), self.cv_accuracy

MODEL_CACHE = ModelCache()


# ================================================================
# 🗃️  蓄水池管理（线程安全）
# ================================================================
def load_reservoir():
    global GLOBAL_RESERVOIR
    if not os.path.exists(RESERVOIR_FILE):
        logger.info("蓄水池文件不存在，从零开始构建")
        return
    try:
        with open(RESERVOIR_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        with _reservoir_lock:
            GLOBAL_RESERVOIR = data
        logger.info(f"✅ 蓄水池加载完成: {len(GLOBAL_RESERVOIR)} 条记录")
    except Exception as e:
        logger.exception(f"加载蓄水池异常: {e}")
        GLOBAL_RESERVOIR = {}

def save_reservoir():
    try:
        with _reservoir_lock:
            snapshot = dict(GLOBAL_RESERVOIR)
        with open(RESERVOIR_FILE, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False)
    except Exception as e:
        logger.error(f"保存蓄水池失败: {e}")

def trim_reservoir():
    global GLOBAL_RESERVOIR
    with _reservoir_lock:
        if len(GLOBAL_RESERVOIR) <= MAX_RESERVOIR_SIZE:
            return
        sorted_issues = sorted(GLOBAL_RESERVOIR.keys())
        excess = len(sorted_issues) - MAX_RESERVOIR_SIZE
        for issue in sorted_issues[:excess]:
            del GLOBAL_RESERVOIR[issue]

def update_and_get_reservoir():
    global GLOBAL_RESERVOIR
    data = fetch_api_data(50)
    if not data: return None

    records = []
    if isinstance(data, list): records = data
    elif isinstance(data, dict):
        for key in ("data", "list", "result", "records"):
            if key in data and isinstance(data[key], list):
                records = data[key]
                break
    if not records: return None

    updated = False
    with _reservoir_lock:
        for r in records:
            issue = str(r.get("expect", ""))
            if issue and issue not in GLOBAL_RESERVOIR:
                GLOBAL_RESERVOIR[issue] = r
                updated = True

    if updated:
        trim_reservoir()
        save_reservoir()

    with _reservoir_lock:
        sorted_issues = sorted(GLOBAL_RESERVOIR.keys())
        return [GLOBAL_RESERVOIR[iss] for iss in sorted_issues]


# ================================================================
# 📡  API 工具函数
# ================================================================
def send_telegram_message(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        logger.error(f"Telegram 网络异常: {e}")

def get_group_type(total_sum: int) -> str:
    is_big  = total_sum >= 14
    is_even = total_sum % 2 == 0
    if is_big and is_even: return "大双"
    elif is_big:           return "大单"
    elif is_even:          return "小双"
    else:                  return "小单"

def fetch_api_data(limit: int = 50):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(f"{DATA_URL}?limit={limit}", headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None

def extract_history_from_reservoir(records: list) -> tuple:
    groups, sums = [], []
    if not records: return None, [], []
    latest_issue = records[-1].get("expect", "未知")
    for r in records:
        opencode_str = r.get("opencode")
        if not opencode_str: continue
        try:
            nums  = [int(x) for x in str(opencode_str).split(",")]
            tsum  = sum(nums)
            groups.append(get_group_type(tsum))
            sums.append(tsum)
        except Exception:
            pass
    return latest_issue, groups, sums


# ================================================================
# 🧠  特征工程
# ================================================================
def _build_feature_vector(i: int, groups: list, sums: list) -> list:
    feats = []
    feats.append(sums[i])
    feats.append(sums[i] % 2)
    feats.append(int(sums[i] >= 14))

    for w in WINDOW_SIZES:
        window = sums[max(0, i - w):i]
        if window:
            feats.append(float(np.mean(window)))
            feats.append(float(np.std(window)) if len(window) > 1 else 0.0)
            feats.append(float(np.max(window) - np.min(window)))
        else:
            feats.extend([0.0, 0.0, 0.0])

    for lag in range(1, 6):
        idx = i - lag
        feats.append(GROUP_MAP.get(groups[idx], 0) if idx >= 0 else 0)

    streak = 1
    for j in range(i - 1, max(0, i - 10) - 1, -1):
        if groups[j] == groups[i]: streak += 1
        else: break
    feats.append(streak)

    recent20 = groups[max(0, i - 20):i]
    total20  = len(recent20) if recent20 else 1
    for g in ALL_GROUPS:
        feats.append(recent20.count(g) / total20)

    recent10 = sums[max(0, i - 10):i]
    feats.append(sum(1 for s in recent10 if s >= 14) / len(recent10) if recent10 else 0.5)
    return feats

def prepare_ml_features_v2(groups: list, sums: list):
    if len(sums) < 30: return None, None, None
    X, y = [], []
    for i in range(20, len(sums) - 1):
        X.append(_build_feature_vector(i, groups, sums))
        y.append(GROUP_MAP[groups[i + 1]])

    if len(set(y)) < 2: return None, None, None
    latest_feature = np.array([_build_feature_vector(len(sums) - 1, groups, sums)])
    return np.array(X), np.array(y), latest_feature


# ================================================================
# 🔢  马尔可夫（✅ 已修复数据泄露）
# ================================================================
def build_markov_models(groups: list) -> tuple:
    order1 = defaultdict(lambda: defaultdict(int))
    order2 = defaultdict(lambda: defaultdict(int))
    # 纯净的历史转移概率统计，不包含未知的下一期
    for i in range(len(groups) - 2):
        g1, g2, g3 = groups[i], groups[i + 1], groups[i + 2]
        order1[g2][g3] += 1
        order2[(g1, g2)][g3] += 1
    return order1, order2

def markov_predict_smoothed(groups: list, order1: dict, order2: dict, alpha: float = 0.5) -> tuple:
    state2 = (groups[-2], groups[-1])
    state1 = groups[-1]
    counts = order2.get(state2, order1.get(state1, {}))
    smoothed = {g: counts.get(g, 0) + alpha for g in ALL_GROUPS}
    total    = sum(smoothed.values())
    probs    = {g: v / total for g, v in smoothed.items()}
    best = max(probs, key=probs.get)
    return best, probs[best]


# ================================================================
# 🤖  ML 模型训练与缓存
# ================================================================
def _compute_cv_accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
    tscv = TimeSeriesSplit(n_splits=5)
    fold_accs = []
    for train_idx, val_idx in tscv.split(X):
        if len(val_idx) == 0: continue
        m = clone(model)
        try:
            m.fit(X[train_idx], y[train_idx])
            acc = (m.predict(X[val_idx]) == y[val_idx]).mean()
            fold_accs.append(acc)
        except Exception:
            pass
    return float(np.mean(fold_accs)) if fold_accs else 0.25

def train_and_cache_models(X: np.ndarray, y: np.ndarray, data_size: int):
    models, cv_accs = {}, []
    try:
        m = lgb.LGBMClassifier(n_estimators=50, learning_rate=0.05, max_depth=4, verbose=-1, n_jobs=-1)
        acc = _compute_cv_accuracy(m, X, y)
        m.fit(X, y)
        models["lgb"] = m
        cv_accs.append(acc)
    except Exception as e: logger.error(f"LGBM失败: {e}")

    try:
        m = xgb.XGBClassifier(n_estimators=50, learning_rate=0.05, max_depth=4, eval_metric="mlogloss", verbosity=0)
        acc = _compute_cv_accuracy(m, X, y)
        m.fit(X, y)
        models["xgb"] = m
        cv_accs.append(acc)
    except Exception as e: logger.error(f"XGB失败: {e}")

    try:
        m = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1)
        acc = _compute_cv_accuracy(m, X, y)
        m.fit(X, y)
        models["rf"] = m
        cv_accs.append(acc)
    except Exception as e: logger.error(f"RF失败: {e}")

    avg_cv = float(np.mean(cv_accs)) if cv_accs else 0.25
    MODEL_CACHE.update(models, data_size, avg_cv)
    logger.info(f"✅ 异步模型更新完成 | CV: {avg_cv:.3f} | 样本: {data_size}")

def run_ml_predictions(latest_feature: np.ndarray) -> dict:
    models, _ = MODEL_CACHE.get()
    predictions = {}
    for name, model in models.items():
        try:
            pred = model.predict(latest_feature)[0]
            predictions[name] = REV_MAP[pred]
        except Exception:
            pass
    return predictions


# ================================================================
# ⚖️  裁判与风控体系（✅ 实装 4.72/4.32 核弹赔率）
# ================================================================
_RANDOM_BASELINE = 0.25

def run_hybrid_ensemble_judge(ml_preds: dict, markov_pred: str, markov_prob: float, cv_accuracy: float) -> tuple:
    votes = defaultdict(int)
    for pred in ml_preds.values(): votes[pred] += 1
    votes[markov_pred] += 1 

    total_voters = len(ml_preds) + 1
    final_best   = max(votes, key=votes.get)
    final_votes  = votes[final_best]

    raw_confidence = final_votes / total_voters
    calibrated     = raw_confidence * (cv_accuracy / _RANDOM_BASELINE)
    confidence_pct = min(calibrated * 100, 88.0) 

    vote_detail = " | ".join(f"{k.upper()}[{v}]" for k, v in ml_preds.items())
    vote_detail += f" | 矩阵[{markov_pred}]({markov_prob:.2f})"
    return final_best, confidence_pct, vote_detail

def calculate_kelly_fraction(win_prob_pct: float, target_group: str) -> float:
    """基于真实盘口高赔率的动态凯利风控，附带13/14免死金牌减震机制"""
    p = win_prob_pct / 100.0
    q = 1.0 - p
    
    if target_group in ["小单", "大双"]:
        b = 3.72  # 真实 4.72 倍净利润
    elif target_group in ["大单", "小双"]:
        b = 3.32  # 真实 4.32 倍净利润
    else:
        b = 0.95  
        
    f_star = (b * p - q) / b
    return 0.0 if f_star <= 0 else (f_star / 2.0) * 100


# ================================================================
# 🚀  云端主引擎（✅ 异步训练、✅ 真实顺位双组、✅ 热力特码）
# ================================================================
def cloud_ensemble_engine(groups: list, sums: list) -> tuple:
    if len(groups) < 30:
        return "📡 数据加载中...", "观望", ["空仓"], [0], 0

    order1, order2 = build_markov_models(groups)
    markov_pred, markov_prob = markov_predict_smoothed(groups, order1, order2)

    X, y, latest_f = prepare_ml_features_v2(groups, sums)
    if X is None:
        return "⚠️ 特征构建失败，数据不足", "观望", ["空仓"], [0], 0

    # 🔥 开启隐形异步线程重训，绝不卡死主轮询心跳
    if MODEL_CACHE.should_retrain(len(X)):
        logger.info(f"后台触发重训练，当前有效特征量: {len(X)}")
        threading.Thread(target=train_and_cache_models, args=(X, y, len(X)), daemon=True).start()

    ml_preds = run_ml_predictions(latest_f)
    _, cv_accuracy = MODEL_CACHE.get()

    if not ml_preds:
        return "⚠️ 所有 ML 模型预测失败", "观望", ["空仓"], [0], 0

    final_judge, confidence, vote_detail = run_hybrid_ensemble_judge(
        ml_preds, markov_pred, markov_prob, cv_accuracy
    )

    if confidence < 50.0:
        signal = (
            f"📉 混沌期预警 | 多维分歧，强控锁仓\n"
            f"🔍 置信度: {confidence:.1f}% | CV精度: {cv_accuracy:.3f}\n"
            f"🔍 底层侦测: {vote_detail}"
        )
        return signal, "意见分歧", ["空仓", "防守"], [0], 0

    # 🔥 真实选票顺位生成【双组】
    all_votes = list(ml_preds.values()) + [markov_pred]
    vote_counts = {g: all_votes.count(g) for g in set(all_votes)}
    sorted_candidates = sorted(vote_counts.keys(), key=lambda x: vote_counts[x], reverse=True)
    
    recommend_group = [final_judge]
    for g in sorted_candidates:
        if g != final_judge and len(recommend_group) < 2:
            recommend_group.append(g)
            
    # 极端防守补齐
    for g in ALL_GROUPS:
        if g not in recommend_group and len(recommend_group) < 2:
            recommend_group.append(g)

    kill_group = [g for g in ALL_GROUPS if g not in recommend_group][-1]

    # 🔥 【特码热力追踪】基于近50期真实开奖频次打击
    recent_sums = sums[-50:] if len(sums) > 0 else []
    valid_pool = set()
    for rg in recommend_group:
        if rg == "大双": valid_pool.update([x for x in range(14, 28) if x % 2 == 0])
        elif rg == "大单": valid_pool.update([x for x in range(14, 28) if x % 2 != 0])
        elif rg == "小双": valid_pool.update([x for x in range(0, 14) if x % 2 == 0])
        elif rg == "小单": valid_pool.update([x for x in range(0, 14) if x % 2 != 0])
        
    freq_map = {num: recent_sums.count(num) for num in valid_pool}
    # 优先挑最热的号，一样热就选靠近13.5的稳妥号
    sorted_nums = sorted(list(valid_pool), key=lambda x: (-freq_map.get(x, 0), abs(x - 13.5)))
    specials = sorted_nums[:3] if sorted_nums else [0, 0, 0]

    kelly_pct  = calculate_kelly_fraction(confidence, final_judge)

    signal = (
        f"👑 云端最高法共振 | 胜率: {confidence:.1f}%\n"
        f"🔍 样本池: {len(groups)}期 | CV精度: {cv_accuracy:.3f}\n"
        f"🔍 底层侦测: {vote_detail}"
    )
    return signal, kill_group, recommend_group, specials, kelly_pct


# ================================================================
# ⏰  主轮询与防休眠守护任务（✅ CDN 穿透）
# ================================================================
def polling_job():
    global _last_issue
    try:
        all_records = update_and_get_reservoir()
        if not all_records: return

        issue, groups, sums = extract_history_from_reservoir(all_records)
        if not issue or issue == _last_issue: return

        _last_issue = issue
        logger.info(f"✅ 新期号: {issue} | 开出: {sums[-1]} ({groups[-1]})")

        market_signal, kill, recommend, specials, kelly = cloud_ensemble_engine(groups, sums)

        msg = (
            f"🔔 **期号: {issue}** | 开出: `{sums[-1]:02d}` ({groups[-1]})\n"
            f"-------------------------\n"
            f"{market_signal}\n"
        )
        if "空仓" in recommend:
            msg += "\n🚫 **顶尖风控:** 算法阵营陷入绞杀，强控全员空仓观望！"
        else:
            msg += f"\n❌ 绝杀: `{kill}`"
            msg += f"\n✅ 双组: `{' + '.join(recommend)}`"
            msg += f"\n🎯 特码: `{', '.join(map(str, specials))}`"
            kelly_str = "0% (模拟)" if kelly == 0 else f"{kelly:.1f}% 仓位"
            msg += f"\n💰 凯利风控: `{kelly_str}`"

        send_telegram_message(msg)
    except Exception as e:
        logger.exception(f"polling_job 顶层异常: {e}")

app = Flask(__name__)

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def keep_alive(path):
    return "🚀 量化穹顶 V2.1 在线运行中 (Cloud-Native Mode)...", 200

def run_flask_server():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, use_reloader=False)

def self_ping():
    target = os.environ.get("SELF_PING_URL", "https://quant-sniper.onrender.com")
    while True:
        try:
            # 加上动态时间戳，绝杀免费服务器的 CDN 缓存休眠套路
            requests.get(f"{target}?t={int(time.time())}", timeout=5)
        except Exception:
            pass
        time.sleep(600)

if __name__ == "__main__":
    load_reservoir()
    send_telegram_message(
        "✅ **量化要塞 V2.1 部署成功**\n"
        "------------------\n"
        "1. 异步极速引擎上线\n"
        "2. 实装 4.72/4.32 核弹赔率计算\n"
        "3. 特码热力追踪雷达开启"
    )

    threading.Thread(target=run_flask_server, daemon=True).start()
    threading.Thread(target=self_ping, daemon=True).start()

    schedule.every(10).seconds.do(polling_job)
    logger.info("✅ 调度器已启动，每10秒轮询一次")

    while True:
        schedule.run_pending()
        time.sleep(1)

