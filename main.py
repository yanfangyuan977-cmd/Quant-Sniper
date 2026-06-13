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
    """缓存已训练模型和 CV 精度，只在数据量增长足够时才重训练。"""

    def __init__(self, retrain_every: int = RETRAIN_EVERY):
        self._lock          = threading.Lock()
        self.models: dict   = {}
        self.last_train_size: int  = 0
        self.retrain_every  = retrain_every
        self.cv_accuracy: float    = 0.25   # 随机基准起点

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
    except json.JSONDecodeError as e:
        logger.error(f"蓄水池文件损坏，清空重建: {e}")
        GLOBAL_RESERVOIR = {}
    except Exception as e:
        logger.exception(f"加载蓄水池异常: {e}")


def save_reservoir():
    try:
        with _reservoir_lock:
            snapshot = dict(GLOBAL_RESERVOIR)
        with open(RESERVOIR_FILE, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False)
    except Exception as e:
        logger.error(f"保存蓄水池失败: {e}")


def trim_reservoir():
    """保留最近 MAX_RESERVOIR_SIZE 期，防止内存无限增长。"""
    global GLOBAL_RESERVOIR
    with _reservoir_lock:
        if len(GLOBAL_RESERVOIR) <= MAX_RESERVOIR_SIZE:
            return
        sorted_issues = sorted(GLOBAL_RESERVOIR.keys())
        excess = len(sorted_issues) - MAX_RESERVOIR_SIZE
        for issue in sorted_issues[:excess]:
            del GLOBAL_RESERVOIR[issue]
    logger.info(f"蓄水池裁剪完成，删除 {excess} 条旧记录")


def update_and_get_reservoir():
    global GLOBAL_RESERVOIR
    data = fetch_api_data(50)
    if not data:
        return None

    records = []
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for key in ("data", "list", "result", "records"):
            if key in data and isinstance(data[key], list):
                records = data[key]
                break

    if not records:
        logger.warning("API 返回数据为空")
        return None

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
        logger.warning("Telegram 凭证未配置，跳过发送")
        return
    url     = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, json=payload, timeout=5)
        if resp.status_code != 200:
            logger.warning(f"Telegram 发送失败: {resp.status_code} {resp.text[:200]}")
    except requests.exceptions.Timeout:
        logger.warning("Telegram 消息发送超时")
    except requests.exceptions.RequestException as e:
        logger.error(f"Telegram 网络异常: {e}")


def get_group_type(total_sum: int) -> str:
    is_big  = total_sum >= 14
    is_even = total_sum % 2 == 0
    if is_big and is_even:      return "大双"
    elif is_big:                return "大单"
    elif is_even:               return "小双"
    else:                       return "小单"


def fetch_api_data(limit: int = 50):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        resp = requests.get(f"{DATA_URL}?limit={limit}", headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        logger.warning(f"API 返回非200状态: {resp.status_code}")
        return None
    except requests.exceptions.Timeout:
        logger.warning("API 请求超时")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"API 连接失败: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"API 请求异常: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"API 响应 JSON 解析失败: {e}")
        return None


def extract_history_from_reservoir(records: list) -> tuple:
    groups, sums = [], []
    if not records:
        return None, [], []
    latest_issue = records[-1].get("expect", "未知")
    for r in records:
        opencode_str = r.get("opencode")
        if not opencode_str:
            continue
        try:
            nums  = [int(x) for x in str(opencode_str).split(",")]
            tsum  = sum(nums)
            groups.append(get_group_type(tsum))
            sums.append(tsum)
        except (ValueError, AttributeError) as e:
            logger.debug(f"解析 opencode 失败: {opencode_str} → {e}")
    return latest_issue, groups, sums


# ================================================================
# 🧠  特征工程（增强版）
# ================================================================

def _build_feature_vector(i: int, groups: list, sums: list) -> list:
    """
    为第 i 个时间步构造特征向量：
      - 原始值与基础衍生（奇偶、大小）
      - 多窗口滑动统计（均值、标准差、极差）
      - Lag 特征（前5期组别编码）
      - 当前类型连续计数
      - 最近20期各类型频率
      - 最近10期大/小比例
    """
    feats = []

    # 原始值与基础衍生
    feats.append(sums[i])
    feats.append(sums[i] % 2)
    feats.append(int(sums[i] >= 14))

    # 滑动窗口统计
    for w in WINDOW_SIZES:
        window = sums[max(0, i - w):i]
        if window:
            feats.append(float(np.mean(window)))
            feats.append(float(np.std(window)) if len(window) > 1 else 0.0)
            feats.append(float(np.max(window) - np.min(window)))
        else:
            feats.extend([0.0, 0.0, 0.0])

    # Lag 特征（前5期组别编码）
    for lag in range(1, 6):
        idx = i - lag
        feats.append(GROUP_MAP.get(groups[idx], 0) if idx >= 0 else 0)

    # 当前类型连续计数（最多回溯10期）
    streak = 1
    for j in range(i - 1, max(0, i - 10) - 1, -1):
        if groups[j] == groups[i]:
            streak += 1
        else:
            break
    feats.append(streak)

    # 最近20期各类型频率
    recent20 = groups[max(0, i - 20):i]
    total20  = len(recent20) if recent20 else 1
    for g in ALL_GROUPS:
        feats.append(recent20.count(g) / total20)

    # 最近10期大/小比例
    recent10 = sums[max(0, i - 10):i]
    feats.append(sum(1 for s in recent10 if s >= 14) / len(recent10) if recent10 else 0.5)

    return feats


def prepare_ml_features_v2(groups: list, sums: list):
    if len(sums) < 30:
        return None, None, None

    X, y = [], []
    for i in range(20, len(sums) - 1):
        X.append(_build_feature_vector(i, groups, sums))
        y.append(GROUP_MAP[groups[i + 1]])

    if len(set(y)) < 2:
        return None, None, None

    latest_feature = np.array([_build_feature_vector(len(sums) - 1, groups, sums)])
    return np.array(X), np.array(y), latest_feature


# ================================================================
# 🔢  马尔可夫（Laplace 平滑版）
# ================================================================

def build_markov_models(groups: list) -> tuple:
    order1 = defaultdict(lambda: defaultdict(int))
    order2 = defaultdict(lambda: defaultdict(int))
    for i in range(len(groups) - 2):
        g1, g2, g3 = groups[i], groups[i + 1], groups[i + 2]
        order1[g2][g3] += 1
        order2[(g1, g2)][g3] += 1
    if len(groups) >= 2:
        order1[groups[-2]][groups[-1]] += 1
    return order1, order2


def markov_predict_smoothed(
    groups: list, order1: dict, order2: dict, alpha: float = 0.5
) -> tuple:
    """
    Laplace 平滑马尔可夫预测。
    返回 (预测组别, 该组别平滑概率)，彻底消除零概率问题。
    """
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
    """时间序列五折交叉验证，返回平均精度（替代硬编码置信度）。"""
    tscv      = TimeSeriesSplit(n_splits=5)
    fold_accs = []
    for train_idx, val_idx in tscv.split(X):
        if len(val_idx) == 0:
            continue
        m = clone(model)
        try:
            m.fit(X[train_idx], y[train_idx])
            acc = (m.predict(X[val_idx]) == y[val_idx]).mean()
            fold_accs.append(acc)
        except Exception as e:
            logger.debug(f"CV fold 训练失败: {e}")
    return float(np.mean(fold_accs)) if fold_accs else 0.25


def train_and_cache_models(X: np.ndarray, y: np.ndarray, data_size: int):
    """训练三个模型，写入缓存，同时记录 CV 精度。"""
    models, cv_accs = {}, []

    # ── LightGBM ──
    try:
        m = lgb.LGBMClassifier(
            n_estimators=50, learning_rate=0.05, max_depth=4,
            verbose=-1, n_jobs=-1
        )
        acc = _compute_cv_accuracy(m, X, y)
        m.fit(X, y)
        models["lgb"] = m
        cv_accs.append(acc)
        logger.info(f"LightGBM CV 精度: {acc:.3f}")
    except Exception as e:
        logger.error(f"LightGBM 训练失败: {e}")

    # ── XGBoost ──
    try:
        m = xgb.XGBClassifier(
            n_estimators=50, learning_rate=0.05, max_depth=4,
            eval_metric="mlogloss", verbosity=0
        )
        acc = _compute_cv_accuracy(m, X, y)
        m.fit(X, y)
        models["xgb"] = m
        cv_accs.append(acc)
        logger.info(f"XGBoost CV 精度: {acc:.3f}")
    except Exception as e:
        logger.error(f"XGBoost 训练失败: {e}")

    # ── Random Forest ──
    try:
        m = RandomForestClassifier(
            n_estimators=50, max_depth=4, random_state=42, n_jobs=-1
        )
        acc = _compute_cv_accuracy(m, X, y)
        m.fit(X, y)
        models["rf"] = m
        cv_accs.append(acc)
        logger.info(f"Random Forest CV 精度: {acc:.3f}")
    except Exception as e:
        logger.error(f"Random Forest 训练失败: {e}")

    avg_cv = float(np.mean(cv_accs)) if cv_accs else 0.25
    MODEL_CACHE.update(models, data_size, avg_cv)
    logger.info(f"✅ 模型缓存更新完成 | 平均 CV 精度: {avg_cv:.3f} | 样本量: {data_size}")


def run_ml_predictions(latest_feature: np.ndarray) -> dict:
    """从缓存模型中读取预测，不触发重训练。"""
    models, _ = MODEL_CACHE.get()
    predictions = {}
    for name, model in models.items():
        try:
            pred = model.predict(latest_feature)[0]
            predictions[name] = REV_MAP[pred]
        except Exception as e:
            logger.warning(f"模型 {name} 预测失败: {e}")
    return predictions


# ================================================================
# ⚖️  融合裁判（真实置信度，彻底告别硬编码数字）
# ================================================================

_RANDOM_BASELINE = 0.25   # 四分类随机猜测的理论上限


def run_hybrid_ensemble_judge(
    ml_preds: dict,
    markov_pred: str,
    markov_prob: float,
    cv_accuracy: float,
) -> tuple:
    """
    基于 CV 精度的校准置信度融合。
    置信度 = (加权票数比) × (CV精度 / 随机基准) × 100，上限 88%。
    """
    votes = defaultdict(int)
    for pred in ml_preds.values():
        votes[pred] += 1
    votes[markov_pred] += 1                          # 马尔可夫参与投票

    total_voters = len(ml_preds) + 1
    final_best   = max(votes, key=votes.get)
    final_votes  = votes[final_best]

    raw_confidence = final_votes / total_voters
    calibrated     = raw_confidence * (cv_accuracy / _RANDOM_BASELINE)
    confidence_pct = min(calibrated * 100, 88.0)    # 上限88%，避免过度自信

    vote_detail = " | ".join(f"{k.upper()}[{v}]" for k, v in ml_preds.items())
    vote_detail += f" | 矩阵[{markov_pred}]({markov_prob:.2f})"

    return final_best, confidence_pct, vote_detail


def calculate_kelly_fraction(win_prob_pct: float, odds: float = 0.95) -> float:
    p      = win_prob_pct / 100.0
    q      = 1.0 - p
    f_star = (odds * p - q) / odds
    return 0.0 if f_star <= 0 else (f_star / 2.0) * 100


def get_dynamic_specials(target_groups: list, all_sums: list) -> list:
    if "空仓" in target_groups:
        return [0, 0, 0]
    pool = {
        "大双": [14, 16, 18, 20, 22, 24, 26],
        "大单": [15, 17, 19, 21, 23, 25, 27],
        "小双": [0,  2,  4,  6,  8, 10, 12],
        "小单": [1,  3,  5,  7,  9, 11, 13],
    }
    local_mean = sum(all_sums[-20:]) / 20.0 if len(all_sums) >= 20 else 13.5
    candidates = []
    for g in target_groups:
        candidates.extend(pool.get(g, []))
    candidates.sort(key=lambda x: abs(x - local_mean + random.uniform(-0.5, 0.5)))
    return candidates[:3]


# ================================================================
# 🚀  云端主引擎
# ================================================================

def cloud_ensemble_engine(groups: list, sums: list) -> tuple:
    if len(groups) < 30:
        return "📡 数据加载中...", "观望", ["空仓"], [0], 0

    # ── 马尔可夫（Laplace 平滑）──
    order1, order2 = build_markov_models(groups)
    markov_pred, markov_prob = markov_predict_smoothed(groups, order1, order2)

    # ── 特征工程 ──
    X, y, latest_f = prepare_ml_features_v2(groups, sums)
    if X is None:
        return "⚠️ 特征构建失败，数据不足", "观望", ["空仓"], [0], 0

    # ── 按需重训练（增量触发）──
    if MODEL_CACHE.should_retrain(len(groups)):
        logger.info(f"触发增量重训练，当前样本量: {len(groups)}")
        train_and_cache_models(X, y, len(groups))

    ml_preds = run_ml_predictions(latest_f)
    _, cv_accuracy = MODEL_CACHE.get()

    if not ml_preds:
        return "⚠️ 所有 ML 模型预测失败", "观望", ["空仓"], [0], 0

    final_judge, confidence, vote_detail = run_hybrid_ensemble_judge(
        ml_preds, markov_pred, markov_prob, cv_accuracy
    )

    # ── 混沌期：置信度过低强制空仓 ──
    if confidence < 50.0:
        signal = (
            f"📉 混沌期预警 | 多维分歧，强控锁仓\n"
            f"🔍 置信度: {confidence:.1f}% | CV精度: {cv_accuracy:.3f}\n"
            f"🔍 底层侦测: {vote_detail}"
        )
        return signal, "意见分歧", ["空仓", "防守"], [0], 0

    # ── 正常决策 ──
    recommend_group = [final_judge]
    for g in ALL_GROUPS:
        if g != final_judge and len(recommend_group) < 2:
            recommend_group.append(g)
    kill_group = [g for g in ALL_GROUPS if g not in recommend_group][-1]
    specials   = get_dynamic_specials(recommend_group, sums)
    kelly_pct  = calculate_kelly_fraction(confidence)

    signal = (
        f"👑 云端最高法共振 | 胜率: {confidence:.1f}%\n"
        f"🔍 样本池: {len(groups)}期 | CV精度: {cv_accuracy:.3f}\n"
        f"🔍 底层侦测: {vote_detail}"
    )
    return signal, kill_group, recommend_group, specials, kelly_pct


# ================================================================
# ⏰  主轮询任务（schedule 驱动，替代 busy-loop）
# ================================================================

def polling_job():
    global _last_issue
    try:
        all_records = update_and_get_reservoir()
        if not all_records:
            logger.warning("蓄水池为空，跳过本轮")
            return

        issue, groups, sums = extract_history_from_reservoir(all_records)
        if not issue or issue == _last_issue:
            return

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


# ================================================================
# 🌐  防休眠 Flask 伪装层
# ================================================================

app = Flask(__name__)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def keep_alive(path):
    return "🚀 量化穹顶引擎在线运行中 (Cloud-Native Mode)...", 200


def run_flask_server():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, use_reloader=False)


def self_ping():
    target = os.environ.get("SELF_PING_URL", "https://quant-sniper.onrender.com")
    while True:
        try:
            requests.get(target, timeout=5)
        except requests.exceptions.RequestException:
            pass
        except Exception as e:
            logger.debug(f"self_ping 非网络异常: {e}")
        time.sleep(600)


# ================================================================
# 🎯  程序入口
# ================================================================

if __name__ == "__main__":
    load_reservoir()
    send_telegram_message(
        "✅ **量化要塞重启成功**\n"
        "------------------\n"
        "数据蓄水池已开启，最高法官裁决系统运转中！"
    )

    threading.Thread(target=run_flask_server, daemon=True).start()
    threading.Thread(target=self_ping, daemon=True).start()

    schedule.every(10).seconds.do(polling_job)
    logger.info("✅ 调度器已启动，每10秒轮询一次")

    while True:
        schedule.run_pending()
        time.sleep(1)
