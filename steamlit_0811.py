# app.py
from typing import List, cast
import io
import numpy as np
import pandas as pd
import streamlit as st

# ====================== 基础设置 ======================
st.set_page_config(page_title="售后日志分析", layout="wide")
st.markdown("# 售后日志分析")
st.caption("上传多个 .txt/.csv 文件 → 合并为一个 DataFrame → 侧边栏选择设备（SN）→ 拒保条件判断 / 维修逻辑条件判断。")

# —— UI 样式：标题对比、更紧凑的侧边栏、卡片命中红字、顶部间距收窄 —— 
st.markdown("""
<style>
/* 强化 H1/H2/H3 的层级差异 + 收窄顶部间距 */
h1, .stMarkdown h1 {
  font-size: 2.2rem; font-weight: 800; letter-spacing: .3px;
  margin: 0.2rem 0 0.6rem !important;  /* 减小底部间距 */
}
h2, .stMarkdown h2 {
  font-size: 1.6rem; font-weight: 750;
  border-left: 6px solid #4F46E5; padding-left: .55rem;
  margin: 0.6rem 0 0.3rem !important;  /* 缩小上下间距 */
}
h3, .stMarkdown h3 {
  font-size: 1.1rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: .08em; color: #374151;
  margin: 0.4rem 0 0.25rem !important;  /* 更紧凑 */
}

/* 顶部说明文字间距 */
.stMarkdown p {
  margin-top: 0.1rem !important;
  margin-bottom: 0.3rem !important;
}

/* Tab 组件上下间距收窄 */
.stTabs [role="tablist"] {
  margin-top: 0rem !important;
  margin-bottom: 0.2rem !important;
}

/* metric 指标上下间距收窄 */
[data-testid="stMetric"] {
  padding-top: 0.1rem !important;
  padding-bottom: 0.1rem !important;
}

/* 侧边栏组件之间更紧凑 */
div[data-testid="stSidebar"] .stTextInput,
div[data-testid="stSidebar"] .stNumberInput,
div[data-testid="stSidebar"] .stSelectbox,
div[data-testid="stSidebar"] .stSlider,
div[data-testid="stSidebar"] .stRadio,
div[data-testid="stSidebar"] .stToggle,
div[data-testid="stSidebar"] .stCaptionContainer {
  margin-bottom: .35rem;
}

/* metric 数值样式 */
[data-testid="stMetricValue"] { font-size: 1.3rem; }
[data-testid="stMetricDelta"] { font-size: .85rem; }

/* —— 命中/前文/后文的单元格样式 —— */
td.card-hit  { color: #B91C1C !important; font-weight: 900 !important; }  /* 命中行红字+加粗 */
td.card-pre  { background: #E7F0FF !important; }
td.card-post { background: #E7F8EF !important; }

/* 固定第一列（“标记”列），继承行样式 */
table tbody tr td:first-child {
  position: sticky; left: 0; z-index: 1; background: inherit; color: inherit;
}

/* 表头小阴影 */
thead tr th { box-shadow: 0 1px 0 rgba(0,0,0,.06); }
            
/* 调整页面整体内容距离顶端的距离 */
.block-container {
    padding-top: 2.2rem !important;  /* 默认大约 6rem，可以改成更小的值 */
}
            
</style>
""", unsafe_allow_html=True)



# ====================== 常量（默认，可被侧边栏覆盖） ======================
DEFAULT_COCA_KEYWORD = "coca"
DEFAULT_DOCA_KEYWORD = "doca"
DEFAULT_PREV_N = 5
DEFAULT_NEXT_N = 1

# ====================== 读文件 & 预处理 ======================
def _robust_read_csv(b: bytes, filename: str) -> pd.DataFrame:
    """
    自动解析：编码 utf-8→gbk→latin-1，分隔符 自动→\t→,→;→|→空白
    """
    encodings = ["utf-8", "gbk", "latin-1"]
    seps = [None, "\t", ",", ";", "|", r"\s+"]
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(
                    io.BytesIO(b),
                    sep=sep,
                    encoding=enc,
                    engine="python",
                    on_bad_lines="skip",
                )
                df["source_file"] = filename
                return df
            except Exception:
                continue
    # 兜底
    df = pd.read_csv(
        io.BytesIO(b),
        sep=r"\s+",
        encoding="latin-1",
        engine="python",
        on_bad_lines="skip",
    )
    df["source_file"] = filename
    return df

@st.cache_data(show_spinner=False)
def read_many(files) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        df = _robust_read_csv(f.read(), f.name)
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out["_row_order"] = np.arange(len(out))
    return out

def infer_column(df: pd.DataFrame, candidates: List[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in cols_lower:
            return cols_lower[c]
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in candidates):
            return c
    return None

def normalize_columns(df: pd.DataFrame, col_sn: str | None, col_err: str | None, col_ts: str | None) -> tuple[pd.DataFrame, list[str]]:
    warns = []
    x = df.copy()
    if col_sn:
        x = x.rename(columns={col_sn: "sn"})
    else:
        x["sn"] = "UNKNOWN"
        warns.append("未找到 sn 列，已填充占位值 'UNKNOWN'。")
    if col_err:
        x = x.rename(columns={col_err: "error"})
    else:
        x["error"] = ""
        warns.append("未找到 error 列，已填充空字符串。")
    if col_ts:
        x = x.rename(columns={col_ts: "timestamp"})
        try:
            x["timestamp"] = pd.to_datetime(x["timestamp"], errors="coerce")
        except Exception:
            warns.append("timestamp 解析失败，已忽略时间排序。")
    x["sn"] = x["sn"].astype(str)
    x["error"] = x["error"].astype(str)
    return x, warns

def demo_data(n_sn=3, rows_per_sn=80, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sns = [f"SN{1000+i}" for i in range(n_sn)]
    rows = []
    for sn in sns:
        for i in range(rows_per_sn):
            ts = pd.Timestamp("2025-06-01") + pd.to_timedelta(i, unit="m")
            if rng.random() < 0.06:
                err = f"device warning: COCA code={rng.integers(100,999)}"
            else:
                err = f"ok message {rng.integers(1000,9999)}"
            rows.append({"timestamp": ts, "sn": sn, "error": err, "_row_order": i, "source_file": "demo",
                         "data_source": "event" if rng.random()<0.7 else "history"})
    return pd.DataFrame(rows)

# ====================== 实用工具 ======================
def _to_int_safe(val) -> int:
    try:
        if pd.isna(val):
            return 0
        if hasattr(val, "item"):
            val = val.item()
        return int(val)
    except Exception:
        try:
            return int(float(val))
        except Exception:
            return 0

def style_card(df_show: pd.DataFrame):
    """
    需要列：['标记','context_pos',...]
    依据“命中/前文/后文”打 class，命中行红字（CSS 控制）；固定第一列
    """
    classes = []
    for _, row in df_show.iterrows():
        mark = str(row.get("标记", ""))
        try:
            pos = float(row.get("context_pos", 0))
        except Exception:
            pos = 0
        if "命中" in mark:
            cname = "card-hit"
        elif pos < 0:
            cname = "card-pre"
        else:
            cname = "card-post"
        classes.append([cname] * df_show.shape[1])

    class_df = pd.DataFrame(classes, columns=df_show.columns, index=df_show.index)
    styler = (
        df_show.style
        .set_table_styles([{"selector": "th", "props": [("position","sticky"),("top","0"),("z-index","1")]}])
        .set_td_classes(class_df)
    )
    if hasattr(styler, "hide_index"):
        styler = styler.hide_index()  # type: ignore[attr-defined]
    else:
        try:
            styler = styler.hide(axis="index")
        except Exception:
            pass
    return styler

def sort_group(g: pd.DataFrame) -> pd.DataFrame:
    sort_cols = []
    if "timestamp" in g.columns:
        sort_cols.append("timestamp")
    sort_cols.append("_row_order")
    return g.sort_values(by=sort_cols, kind="stable").reset_index(drop=True)

def extract_cards_for_sn(g: pd.DataFrame, keyword: str, prev_n: int, next_n: int) -> list[pd.DataFrame]:
    """
    每个命中行生成一张卡片（独立窗口，不合并），返回 DataFrame 列表。
    列含：is_hit, context_pos, card_id 等。
    """
    g_sorted = sort_group(g)
    hits = g_sorted.index[g_sorted["error"].str.contains(keyword, case=False, na=False)].tolist()
    cards = []
    n = len(g_sorted)
    for i, hit_idx in enumerate(hits, start=1):
        s = max(0, hit_idx - prev_n)
        e = min(n - 1, hit_idx + next_n)
        block = g_sorted.iloc[s:e+1].copy()
        block["card_id"] = i
        offsets = np.arange(s, e+1) - hit_idx
        block["context_pos"] = offsets
        block["is_hit"] = (block.index == hit_idx).astype(int)
        cards.append(block.reset_index(drop=True))
    return cards

def _first_last_hit_times(cards: list[pd.DataFrame]) -> tuple[str, str]:
    """从卡片列表提取首次/末次命中时间字符串"""
    times = []
    for card_df in cards:
        hit = card_df.loc[card_df["is_hit"] == 1]
        if "timestamp" in hit.columns and not hit.empty and pd.notnull(hit["timestamp"].iloc[0]):
            times.append(hit["timestamp"].iloc[0])
    if times:
        return str(min(times)), str(max(times))
    return "—", "—"

def _history_long_gap_rows(g: pd.DataFrame, months: int = 6) -> list[tuple[pd.Series, pd.Series, pd.Timedelta]]:
    """
    检查 data_source=='history' 的记录是否存在相邻两条时间差 > months（按 30*months 天近似）。
    """
    if "data_source" not in g.columns:
        return []

    h = g[g["data_source"].astype(str).str.lower() == "history"].copy()
    if h.empty:
        return []

    # 统一时间列：优先 'timestamp'，否则 'time'
    if "timestamp" in h.columns:
        tcol = "timestamp"
    elif "time" in h.columns:
        tcol = "time"
    else:
        return []

    h["_ts"] = pd.to_datetime(h[tcol], errors="coerce")
    h = h.dropna(subset=["_ts"]).sort_values(
        by=["_ts", "_row_order"] if "_row_order" in h.columns else ["_ts"]
    ).reset_index(drop=True)

    if len(h) < 2:
        return []

    threshold = pd.Timedelta(days=30 * months)  # 近似月份
    res: list[tuple[pd.Series, pd.Series, pd.Timedelta]] = []

    for i in range(1, len(h)):
        row_prev = cast(pd.Series, h.iloc[i - 1, :])
        row_next = cast(pd.Series, h.iloc[i, :])
        t0 = row_prev["_ts"]
        t1 = row_next["_ts"]
        delta: pd.Timedelta = t1 - t0  # type: ignore[assignment]
        if pd.isna(delta):
            continue
        if delta > threshold:
            res.append((row_prev, row_next, delta))

    return res

# ====================== 侧边栏：上传 & 设备选择 & 参数 ======================
with st.sidebar:
    st.header("📥 上传")
    files = st.file_uploader(
        "上传多个 .txt / .csv 文件（可多选）",
        type=["txt", "csv"],
        accept_multiple_files=True,
    )
    st.markdown("---")
    demo = st.toggle("使用演示数据（忽略上传）", value=False)

# 读取数据
if demo:
    raw_df = demo_data()
else:
    raw_df = read_many(files)

if raw_df.empty:
    st.info("请在左侧上传文件，或开启演示数据。")
    st.stop()

# 自动识别列（sn / error / time→timestamp）
auto_sn = infer_column(raw_df, ["sn"])
auto_err = infer_column(raw_df, ["error", "err", "errmsg", "message", "msg"])
auto_ts = infer_column(raw_df, ["timestamp", "time", "datetime", "date", "ts"])
df, warns = normalize_columns(raw_df, auto_sn, auto_err, auto_ts)
if warns:
    for w in warns:
        st.warning(w)

# 侧边栏：SN 选择 + 判断参数（更紧凑）
with st.sidebar:
    st.header("🔌 选择设备（SN）")
    all_sns = sorted(df["sn"].astype(str).unique().tolist())
    q = st.text_input("搜索 SN（支持模糊匹配）", value="", placeholder="输入 SN 关键字…")
    filtered = [s for s in all_sns if q.lower() in s.lower()] if q else all_sns
    if q and not filtered:
        st.info("未匹配到 SN，已显示全部。")
        filtered = all_sns
    sn_pick = st.radio("点击 SN 查看页面：", filtered, index=0, label_visibility="visible")

    st.markdown("---")
    st.header("⚙️ 判断参数")

    # 常用：关键词 + History 阈值（月）
    cola, colb = st.columns(2)
    with cola:
        coca_kw = st.text_input("COCA 关键字", value=DEFAULT_COCA_KEYWORD, placeholder="如：coca", key="kw_coca")
    with colb:
        doca_kw = st.text_input("DOCA 关键字", value=DEFAULT_DOCA_KEYWORD, placeholder="如：doca", key="kw_doca")

    history_months = st.number_input(
        "History 相邻间隔阈值（月）",
        min_value=1, max_value=36, value=6, step=1,
        help="用于“History 搁置检查”，相邻两条超过该月数则提示"
    )
    st.session_state["history_months"] = int(history_months)

    # 高级设置：上下文窗口行数
    with st.expander("高级设置：上下文窗口行数", expanded=False):
        ca1, ca2, da1, da2 = st.columns(4)
        with ca1:
            coca_prev = st.number_input("COCA 前置行", min_value=0, max_value=100, value=DEFAULT_PREV_N, step=1, key="prev_coca")
        with ca2:
            coca_next = st.number_input("COCA 后置行", min_value=0, max_value=100, value=DEFAULT_NEXT_N, step=1, key="next_coca")
        with da1:
            doca_prev = st.number_input("DOCA 前置行", min_value=0, max_value=100, value=DEFAULT_PREV_N, step=1, key="prev_doca")
        with da2:
            doca_next = st.number_input("DOCA 后置行", min_value=0, max_value=100, value=DEFAULT_NEXT_N, step=1, key="next_doca")

# ====================== 单 SN 页面（两级 Tab） ======================
g = df[df["sn"] == sn_pick].copy()
st.markdown(f"## {sn_pick}")
st.markdown("### 判断结果")

# 顶层分组：拒保 & 维修 & 原始数据
top_tabs = st.tabs(["拒保条件判断", "维修逻辑条件判断", "原始数据"])

# ---------- 拒保条件判断（COCA / DOCA / POV-PUV / History） ----------
with top_tabs[0]:
    rej_tabs = st.tabs(["COCA", "DOCA", "POV/PUV 次数", "History 搁置检查"])

    # ---------- COCA ----------
    with rej_tabs[0]:
        st.markdown("### COCA 判断结果")
        coca_cards = extract_cards_for_sn(g, keyword=coca_kw, prev_n=int(coca_prev), next_n=int(coca_next))
        first_t, last_t = _first_last_hit_times(coca_cards)

        c1, c2, c3 = st.columns(3)
        c1.metric("COCA 事件数", len(coca_cards))
        c2.metric("首次命中时间", first_t)
        c3.metric("末次命中时间", last_t)

        num_events = len(coca_cards)
        state_key = f"card_idx_{sn_pick}_COCA"
        if state_key not in st.session_state:
            st.session_state[state_key] = 1

        nav_l, nav_c, nav_r = st.columns([1, 3, 1])
        with nav_l:
            prev_disabled = st.session_state[state_key] <= 1 or num_events == 0
            if st.button("⬅️ 上一事件", disabled=prev_disabled, key=f"prev_{state_key}"):
                st.session_state[state_key] = max(1, st.session_state[state_key] - 1)
        with nav_c:
            if num_events > 0:
                new_idx = st.slider("跳转事件序号", 1, num_events, st.session_state[state_key], key=f"slider_{state_key}")
                if new_idx != st.session_state[state_key]:
                    st.session_state[state_key] = new_idx
            else:
                st.write("（无事件可跳转）")
        with nav_r:
            next_disabled = st.session_state[state_key] >= num_events or num_events == 0
            if st.button("下一事件 ➡️", disabled=next_disabled, key=f"next_{state_key}"):
                st.session_state[state_key] = min(num_events, st.session_state[state_key] + 1)

        if num_events > 0:
            all_cards_df = pd.concat(
                [c.drop(columns=["is_hit"], errors="ignore").assign(card_id=_to_int_safe(c["card_id"].iat[0])) for c in coca_cards],
                ignore_index=True,
            )
            st.download_button(
                "⬇️ 下载本设备全部 COCA 事件（CSV）",
                data=all_cards_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{sn_pick}_coca_events_all.csv",
                mime="text/csv",
                key=f"dl_all_coca_{sn_pick}"
            )

        if not coca_cards:
            st.info("未发现包含 COCA 的行。")
        else:
            st.caption(f"共发现 **{num_events}** 个 COCA 事件。使用上方控件快速跳转。")
            current_card_id = st.session_state[state_key] if num_events > 0 else None
            for card_df in coca_cards:
                card_id = _to_int_safe(card_df["card_id"].iat[0])
                hit_row = card_df.loc[card_df["is_hit"] == 1]
                hit_time = hit_row["timestamp"].iloc[0] if "timestamp" in hit_row.columns and not hit_row.empty else None
                hit_msg = hit_row["error"].iloc[0] if not hit_row.empty else ""
                title = f"🧩 事件 #{card_id}"
                subtitle = f"命中: {str(hit_time) if pd.notnull(hit_time) else '无时间'} ｜ {hit_msg}"
                expanded_flag = (card_id == current_card_id)
                with st.expander(f"{title} ｜ {subtitle}", expanded=expanded_flag):
                    # —— 将 COCA 卡片里渲染数据表格的部分替换为下面这段（保留你原有的 show 构建逻辑与下载按钮）——
                    display_cols = [c for c in ["timestamp", "sn", "error", "source_file", "_row_order", "context_pos"] if c in card_df.columns]
                    extra_cols = [c for c in card_df.columns if c not in display_cols + ["is_hit", "card_id"]]
                    show = card_df[display_cols + extra_cols].copy()
                    show.insert(0, "标记", np.where(card_df["is_hit"] == 1, "★ 命中", np.where(card_df["context_pos"] < 0, "↑ 前文", "↓ 后文")))

                    # 仅使用 Streamlit 内置展示：st.dataframe + pandas Styler（命中/包含 COCA 的 error 字体红色）
                    _coca_mask = card_df.get("error", pd.Series([], dtype=str)).str.contains(coca_kw, case=False, na=False)

                    def _style_error_col(col: pd.Series):
                        if col.name != "error":
                            return [""] * len(col)
                        return ["color: #B91C1C; font-weight: 900" if bool(m) else "" for m in _coca_mask]

                    styler = show.style.apply(_style_error_col, axis=0)
                    try:
                        styler = styler.hide_index()  # type: ignore[attr-defined]
                    except Exception:
                        try:
                            styler = styler.hide(axis="index")
                        except Exception:
                            pass

                    st.dataframe(styler, use_container_width=True, height=300)
                    csv_card = card_df.drop(columns=["is_hit"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        f"下载事件 #{card_id}（CSV）",
                        data=csv_card,
                        file_name=f"{sn_pick}_coca_event_{card_id}.csv",
                        mime="text/csv",
                        key=f"dl_coca_{sn_pick}_{card_id}"
                    )

    # ---------- DOCA ----------
    with rej_tabs[1]:
        st.markdown("### DOCA 判断结果")
        doca_cards = extract_cards_for_sn(g, keyword=doca_kw, prev_n=int(doca_prev), next_n=int(doca_next))
        first_t, last_t = _first_last_hit_times(doca_cards)

        c1, c2, c3 = st.columns(3)
        c1.metric("DOCA 事件数", len(doca_cards))
        c2.metric("首次命中时间", first_t)
        c3.metric("末次命中时间", last_t)

        num_events = len(doca_cards)
        state_key = f"card_idx_{sn_pick}_DOCA"
        if state_key not in st.session_state:
            st.session_state[state_key] = 1

        nav_l, nav_c, nav_r = st.columns([1, 3, 1])
        with nav_l:
            prev_disabled = st.session_state[state_key] <= 1 or num_events == 0
            if st.button("⬅️ 上一事件", disabled=prev_disabled, key=f"prev_{state_key}"):
                st.session_state[state_key] = max(1, st.session_state[state_key] - 1)
        with nav_c:
            if num_events > 0:
                new_idx = st.slider("跳转事件序号", 1, num_events, st.session_state[state_key], key=f"slider_{state_key}")
                if new_idx != st.session_state[state_key]:
                    st.session_state[state_key] = new_idx
            else:
                st.write("（无事件可跳转）")
        with nav_r:
            next_disabled = st.session_state[state_key] >= num_events or num_events == 0
            if st.button("下一事件 ➡️", disabled=next_disabled, key=f"next_{state_key}"):
                st.session_state[state_key] = min(num_events, st.session_state[state_key] + 1)

        if num_events > 0:
            all_cards_df = pd.concat(
                [c.drop(columns=["is_hit"], errors="ignore").assign(card_id=_to_int_safe(c["card_id"].iat[0])) for c in doca_cards],
                ignore_index=True,
            )
            st.download_button(
                "⬇️ 下载本设备全部 DOCA 事件（CSV）",
                data=all_cards_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{sn_pick}_doca_events_all.csv",
                mime="text/csv",
                key=f"dl_all_doca_{sn_pick}"
            )

        if not doca_cards:
            st.info("未发现包含 DOCA 的行。")
        else:
            st.caption(f"共发现 **{num_events}** 个 DOCA 事件。使用上方控件快速跳转。")
            current_card_id = st.session_state[state_key] if num_events > 0 else None
            for card_df in doca_cards:
                card_id = _to_int_safe(card_df["card_id"].iat[0])
                hit_row = card_df.loc[card_df["is_hit"] == 1]
                hit_time = hit_row["timestamp"].iloc[0] if "timestamp" in hit_row.columns and not hit_row.empty else None
                hit_msg = hit_row["error"].iloc[0] if not hit_row.empty else ""
                title = f"🧩 事件 #{card_id}"
                subtitle = f"命中: {str(hit_time) if pd.notnull(hit_time) else '无时间'} ｜ {hit_msg}"
                expanded_flag = (card_id == current_card_id)
                with st.expander(f"{title} ｜ {subtitle}", expanded=expanded_flag):
                    display_cols = [c for c in ["timestamp", "sn", "error", "source_file", "_row_order", "context_pos"] if c in card_df.columns]
                    extra_cols = [c for c in card_df.columns if c not in display_cols + ["is_hit", "card_id"]]
                    show = card_df[display_cols + extra_cols].copy()
                    show.insert(0, "标记", np.where(card_df["is_hit"] == 1, "★ 命中", np.where(card_df["context_pos"] < 0, "↑ 前文", "↓ 后文")))
                    try:
                        styler = style_card(show)
                        st.markdown(styler.to_html(), unsafe_allow_html=True)
                    except Exception:
                        st.dataframe(show, use_container_width=True, height=300)
                    csv_card = card_df.drop(columns=["is_hit"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        f"下载事件 #{card_id}（CSV）",
                        data=csv_card,
                        file_name=f"{sn_pick}_doca_event_{card_id}.csv",
                        mime="text/csv",
                        key=f"dl_doca_{sn_pick}_{card_id}"
                    )

    # ---------- POV / PUV 次数 ----------
    with rej_tabs[2]:
        st.markdown("### POV / PUV 次数")
        err_col = "error" if "error" in g.columns else None
        if not err_col:
            st.info("未找到 error 列。")
        else:
            pov_mask = g[err_col].str.contains("pov", case=False, na=False)
            puv_mask = g[err_col].str.contains("puv", case=False, na=False)
            pov_rows = g[pov_mask].copy()
            puv_rows = g[puv_mask].copy()

            colA, colB = st.columns(2)
            colA.metric("POV 次数", len(pov_rows))
            colB.metric("PUV 次数", len(puv_rows))

            st.subheader("POV 命中行")
            st.dataframe(pov_rows, use_container_width=True, height=260)
            st.download_button(
                "下载 POV 命中行（CSV）",
                data=pov_rows.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{sn_pick}_POV_rows.csv",
                mime="text/csv",
                key=f"dl_pov_{sn_pick}"
            )

            st.subheader("PUV 命中行")
            st.dataframe(puv_rows, use_container_width=True, height=260)
            st.download_button(
                "下载 PUV 命中行（CSV）",
                data=puv_rows.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{sn_pick}_PUV_rows.csv",
                mime="text/csv",
                key=f"dl_puv_{sn_pick}"
            )

    # ---------- History 搁置检查 ----------
    with rej_tabs[3]:
        hm = int(st.session_state.get("history_months", 6))
        st.markdown(f"### History 搁置检查（> {hm} 个月）")
        gaps = _history_long_gap_rows(g, months=hm)
        if not gaps:
            st.success("未发现超过阈值的 history 相邻记录间隔。")
        else:
            st.warning(f"发现 **{len(gaps)}** 处超过 {hm} 个月的间隔：")
            for idx, (row0, row1, delta) in enumerate(gaps, start=1):
                st.markdown(f"**间隔 #{idx}**：{delta}  （{row0.get('timestamp', '—')} → {row1.get('timestamp', '—')}）")
                show0 = pd.DataFrame([row0])
                show1 = pd.DataFrame([row1])
                st.write("上一条记录：")
                st.dataframe(show0, use_container_width=True, height=120)
                st.write("下一条记录：")
                st.dataframe(show1, use_container_width=True, height=120)

# ---------- 维修逻辑条件判断（预留 3 个 Tab） ----------
with top_tabs[1]:
    svc_tabs = st.tabs(["充高放低（预留）", "采样异常（预留）", "是否均衡（预留）"])

    with svc_tabs[0]:
        st.markdown("### 充高放低（预留）")
        st.info("维修逻辑 · 充高放低：算法/指标待接入。此处先占位。")

    with svc_tabs[1]:
        st.markdown("### 采样异常（预留）")
        st.info("维修逻辑 · 采样异常：算法/指标待接入。此处先占位。")

    with svc_tabs[2]:
        st.markdown("### 是否均衡（预留）")
        st.info("维修逻辑 · 是否均衡：算法/指标待接入。此处先占位。")

# ---------- 原始数据 ----------
with top_tabs[2]:
    st.markdown("### 原始数据（前 200 行）")
    st.dataframe(g.head(200), use_container_width=True, height=320)
