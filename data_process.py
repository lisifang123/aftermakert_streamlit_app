"""
电池/系统日志解析与清洗（等价重构版）
- 保持原始逻辑/输出字段不变的前提下做结构化与健壮性改进
- 去重/复用：统一常量、正则与通用工具；避免重复实现
- 兼容性：同时兼容 `Txt_File_Name`/`txt_file_name` 与 `Data_Source`/`data_source`
- 健壮性：__main__ 中的空结果保护；更明确的日志
"""
from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chardet
import pandas as pd
from tqdm import tqdm

# =========================
# 日志设置
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# =========================
# 常量与正则模式（集中定义，避免函数内重复编译）
# =========================
TIMESTAMP_RE = re.compile(
    r"\b(\d{2}|\d{4})-(\d{2}|\d{1})-(\d{2}|\d{1}) (\d{2}|\d{1}):(\d{2}|\d{1}):(\d{2}|\d{1})\b"
)
TIME_WORD_RE = re.compile(r"\btime\b", re.IGNORECASE)
DATA_HISTORY_RE = re.compile(r"\bdata\b.*\bhistory\b", re.IGNORECASE)
DATA_EVENT_RE = re.compile(r"\bdata\b.*\bevent\b", re.IGNORECASE)
CMD_COMPLETED_RE = re.compile(r"\bcommand\b.*\bcompleted\b", re.IGNORECASE)
CMD_FAIL_RE = re.compile(r"\bcommand\b.*\bfail\b", re.IGNORECASE)
SEPARATOR_LINE_RE = re.compile(r"^[-_]{4,}\s*$")
SEPARATOR_LINE_MARKERS = ("------", "______")  # 兼容旧逻辑

BATTERY_TABLE_HEADER_RE = re.compile(r"^\s*battery\s+\bvolt\b", re.IGNORECASE)

# =========================
# 路径与读取
# =========================

def find_all_file_paths(root_folder: str | Path) -> List[str]:
    root = Path(root_folder)
    return [str(p) for p in root.rglob("*.txt")]


def detect_file_encoding(path: str) -> str:
    with open(path, "rb") as f:
        content_binary = f.read()
        enc = chardet.detect(content_binary).get("encoding") or "utf-8"
    return enc


def read_text_lines(path: str, encoding: Optional[str] = None) -> List[str]:
    enc = encoding or detect_file_encoding(path)
    with open(path, "r", encoding=enc, errors="ignore") as f:
        return f.readlines()

# =========================
# 通用小工具
# =========================

def normalize_key(raw: str) -> str:
    key = re.sub(r"\.\s*", "_", raw.lower())
    key = "_".join(key.split())
    return key


def split_kv_line(line: str) -> Optional[Tuple[str, str]]:
    # 仅按第一个冒号（或中文冒号）切分，值中允许继续含冒号（如 22:43:45）
    parts = re.split(r"[:：]", line, maxsplit=1)
    if len(parts) < 2:
        return None
    key = normalize_key(parts[0].strip())
    val = parts[1].strip()
    return key, val


def year_standardization(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    row_split = re.split(r"[-:\s]+", s)
    if len(row_split) < 3:
        return None
    try:
        year = int(row_split[0])
    except ValueError:
        return None
    return ("20" + s) if year < 100 else s


def classify_source_from_path(path: str) -> Optional[str]:
    lower = path.lower()
    if "history" in lower:
        return "history"
    if "event" in lower:
        return "event"
    return None

# =========================
# 解析：General Meta（仅文件名与数据源）
# =========================

def parse_general_meta(txt_path: str, lines: List[str]) -> Dict[str, Optional[str]]:
    """只解析通用元信息：Txt_File_Name / Data_Source。
    Data_Source 优先从路径推断；若为空，再从正文里的 history/event 线索补判一次。
    """
    meta: Dict[str, Optional[str]] = {
        "Txt_File_Name": os.path.basename(txt_path),
        "Data_Source": classify_source_from_path(txt_path),
    }
    if meta.get("Data_Source") is None:
        for line in lines:
            if DATA_HISTORY_RE.search(line):
                meta["Data_Source"] = "history"
                break
            if DATA_EVENT_RE.search(line):
                meta["Data_Source"] = "event"
                break
    return meta

# =========================
# 解析：Info / Stat 区块为两个 dict（对外复用）
# =========================

def parse_info_stat_sections(lines: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    info_dict: Dict[str, str] = {}
    stat_dict: Dict[str, str] = {}

    i = 0
    while i < len(lines):
        line = lines[i]
        lower = line.lower()

        # 命中“区块头”：含 info 或 stat 的行
        if ("info" in lower) or ("stat" in lower):
            info_pos = lower.find("info") if "info" in lower else 10**9
            stat_pos = lower.find("stat") if "stat" in lower else 10**9
            target = "info" if info_pos < stat_pos else "stat"

            j = i + 1
            while j < len(lines):
                # 到达“命令完成”或明显的分隔线就停止本段收集
                if CMD_COMPLETED_RE.search(lines[j]) or SEPARATOR_LINE_RE.match(lines[j]) or any(
                    sep in lines[j] for sep in SEPARATOR_LINE_MARKERS
                ):
                    break
                kv = split_kv_line(lines[j])
                if kv:
                    k, v = kv
                    k = normalize_key(k)
                    if target == "info":
                        info_dict[k] = v
                    else:
                        stat_dict[k] = v
                j += 1
            i = j
        else:
            i += 1

    return info_dict, stat_dict

# =========================
# 解析：基于完整时间戳的 Battery 短列表格
# =========================

def _is_separator_or_completed(line: str) -> bool:
    if CMD_COMPLETED_RE.search(line):
        return True
    if SEPARATOR_LINE_RE.match(line):
        return True
    return False


def parse_battery_short_blocks(lines: List[str], start_idx: int) -> Optional[pd.DataFrame]:
    """从包含 Time 的行 start_idx 开始，向下定位 Battery 表头并解析电池明细行；
    在进入表头之前，围绕 Time 行上下采集 K:V（如 Item Index/Voltage/...），
    并在返回的 DataFrame 中将这些 K:V 列以标量广播到每一行。
    """
    # ---------- 先围绕 Time 行收集头部 K:V ----------
    header_kv: Dict[str, str] = {}

    # 当前 Time 行（通常是 "Time : xxxx"）
    if 0 <= start_idx < len(lines):
        kv = split_kv_line(lines[start_idx])
        if kv:
            k, v = kv
            header_kv[normalize_key(k)] = v  # 'time'

    # 向下：直到 Battery 表头 / 命令结束 / 分隔线
    j = start_idx + 1
    while j < len(lines):
        line = lines[j]
        if _is_separator_or_completed(line) or BATTERY_TABLE_HEADER_RE.search(line):
            break
        kv = split_kv_line(line)
        if kv:
            k, v = kv
            header_kv[normalize_key(k)] = v
        j += 1

    # 向上最多回溯 20 行：跳过空行与 '@'，直到遇到分隔线或非 K:V 为止
    up = start_idx - 1
    up_limit = max(0, start_idx - 20)
    while up >= up_limit:
        line = lines[up]
        if _is_separator_or_completed(line):
            break
        s = line.strip()
        if s == "" or s == "@":
            up -= 1
            continue  # 跳过空行/@，继续向上看（这样能抓到 Item Index）
        kv = split_kv_line(line)
        if kv:
            k, v = kv
            nk = normalize_key(k)
            if nk not in header_kv:  # 不覆盖 Time 行同名键
                header_kv[nk] = v
        else:
            break  # 非 K:V 终止
        up -= 1

    # ---------- 原有逻辑：向下定位 Battery 表头 ----------
    j = start_idx + 1
    header_found = False
    columns: List[str] = []

    while j < len(lines):
        line = lines[j]
        if CMD_COMPLETED_RE.search(line):
            return None
        if _is_separator_or_completed(line) or not line.strip():
            j += 1
            continue
        if ("battery" in line.lower()) and (":" not in line) and ("：" not in line):
            columns = re.sub(r"base\s*state", "base_state", line, flags=re.IGNORECASE)
            columns = re.sub(r"\.\s*", "_", columns.lower()).strip().split()
            header_found = True
            break
        j += 1

    if not header_found:
        return None

    # ---------- 原有逻辑：逐行解析电池明细 ----------
    info_battery = {col: [] for col in columns}
    for k in range(j + 1, len(lines)):
        row = lines[k]
        if _is_separator_or_completed(row):
            break
        row_list = row.strip().split()
        if len(row_list) not in (len(columns), len(columns) + 1):
            continue

        if re.search(r"\bpermanent\b", row, re.IGNORECASE) and re.search(r"\bprotection\b", row, re.IGNORECASE):
            plus = False
            for p, c in enumerate(columns):
                if p < len(row_list) and re.search(r"\bpermanent\b", row_list[p], re.IGNORECASE):
                    info_battery[c].append(row_list[p] + " " + (row_list[p + 1] if p + 1 < len(row_list) else ""))
                    plus = True
                    continue
                info_battery[c].append(row_list[p] if not plus else (row_list[p + 1] if p + 1 < len(row_list) else ""))
        else:
            for p, q in enumerate(row_list[: len(columns)]):
                info_battery[columns[p]].append(q)

    df = pd.DataFrame.from_dict(info_battery, orient="columns")

    # ---------- 广播头部 K:V 到每一行 ----------
    if not df.empty and header_kv:
        for k, v in header_kv.items():
            df[k] = v

    return df

# =========================
# Battery Total（短表合并后）转宽表
# =========================

def battery_total_to_wide(
    df_battery_total: pd.DataFrame,
    *,
    max_cells: int = 15,  # 注意：0-14共15个电芯，这里改为15（原16可能多余）
    series_prefix: str = "Bat_",
    fill_value: str = "",
) -> pd.DataFrame:
    if df_battery_total is None or df_battery_total.empty:
        return df_battery_total

    df = df_battery_total.copy()

    # 必须有 Time 列
    if "Time" not in df.columns:
        raise ValueError("battery_total_to_wide 需要存在 'Time' 列。")

    # 兼容大小写不同的元信息列
    key_cols: List[str] = []
    for c in ("Txt_File_Name", "txt_file_name"):
        if c in df.columns and c not in key_cols:
            key_cols.append(c)
    for c in ("Data_Source", "data_source"):
        if c in df.columns and c not in key_cols:
            key_cols.append(c)
    key_cols.append("Time")

    id_col = next((c for c in ("battery", "b_id", "id", "cell") if c in df.columns), None)
    if id_col is None:
        raise ValueError("未找到电池编号列（期望 'battery' 或 'b_id'/'id'/'cell'）。")

    # 电芯级候选（只对这些展开）
    cell_cols_canonical = {
        "volt",
        "curr",
        "tempr",
        "base_state",
        "volt_state",
        "curr_state",
        "temp_state",
        "tempr_state",
        "coulomb",
        "soc",
        "volt_st",
        "temp_st",
        "base_st",
        "ase_st",
    }
    cell_cols_all = [c for c in df.columns if c in cell_cols_canonical]
    if not cell_cols_all:
        return df.drop_duplicates(subset=key_cols)

    # 指标映射
    SPECIAL_MAP = {
        "soc": "SOC",
        "coulomb": "SOC",
        "tempr": "Tempr",
        "volt": "Volt",
        "volt_state": "State",
        "volt_st": "State",
        "temp_state": "Temp_State",
        "tempr_state": "Temp_State",
        "temp_st": "Temp_State",
        "base_state": "BaseState",
        "base_st": "BaseState",
        "ase_st": "BaseState",
        "curr_state": "Curr_State",
    }

    def _pretty_metric(metric: str) -> str:
        k = metric.lower()
        if k in SPECIAL_MAP:
            return SPECIAL_MAP[k]
        return metric[:1].upper() + metric[1:]

    # 展开前去重：按“映射后的名字”唯一
    preference = [
        "soc",
        "coulomb",
        "tempr_state",
        "temp_state",
        "temp_st",
        "volt_state",
        "volt_st",
        "base_state",
        "base_st",
        "ase_st",
        "curr_state",
        "volt",
        "curr",
        "tempr",
    ]
    seen_pretty = set()
    cell_cols: List[str] = []
    for name in preference:
        if name in cell_cols_all:
            pretty = _pretty_metric(name)
            if pretty not in seen_pretty:
                seen_pretty.add(pretty)
                cell_cols.append(name)
    for name in cell_cols_all:
        pretty = _pretty_metric(name)
        if pretty not in seen_pretty:
            seen_pretty.add(pretty)
            cell_cols.append(name)

    # pack 级字段（不展开）
    exclude = set(key_cols + [id_col] + cell_cols)
    pack_cols = [c for c in df.columns if c not in exclude]

    # 去重到单一电芯行
    slim = df[key_cols + [id_col] + cell_cols].copy()
    slim = slim.sort_values(key_cols + [id_col]).drop_duplicates(
        subset=key_cols + [id_col], keep="last"
    )

    # 透视
    pivot = slim.pivot_table(index=key_cols, columns=id_col, values=cell_cols, aggfunc="first")

    # 展平列名：核心修改处（将0-14转为1-15）
    def _to_int_safe(x):
        try:
            return int(str(x).strip())
        except Exception:
            return None

    col_pairs = []
    for metric, bid in pivot.columns:
        pretty = _pretty_metric(metric)
        # 原始bid是0-14，转换为1-15
        original_bid = _to_int_safe(bid)
        if original_bid is not None:
            converted_bid = original_bid + 1  # 关键：0→1，14→15
        else:
            converted_bid = bid  # 非数字ID保持不变（兼容异常情况）
        # 排序键用原始ID（保证0-14的顺序对应1-15）
        sort_key = original_bid if original_bid is not None else 10**9
        # 列名用转换后的ID
        col_pairs.append((sort_key, f"{series_prefix}{pretty}_{converted_bid}", (metric, bid)))

    # 按原始ID排序（保证0-14的顺序）
    col_pairs.sort(key=lambda x: x[0])
    ordered_cols = [cp[2] for cp in col_pairs]
    new_names = [cp[1] for cp in col_pairs]
    pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(ordered_cols))
    pivot.columns = new_names

    # 兜底去重
    if pd.Index(pivot.columns).duplicated(keep="first").any():
        pivot = pivot.loc[:, ~pd.Index(pivot.columns).duplicated(keep="first")]

    # 补齐目标列：按1-15生成（与转换后一致）
    target_cols = []
    for pretty in sorted(seen_pretty, key=str.lower):
        # 生成1到max_cells（15）的列名
        for i in range(1, max_cells + 1):
            target_cols.append(f"{series_prefix}{pretty}_{i}")
    target_cols = list(dict.fromkeys(target_cols))  # 去重
    # 补充缺失的列
    for c in target_cols:
        if c not in pivot.columns:
            pivot[c] = fill_value
    pivot = pivot.reindex(columns=target_cols)

    # 合并 pack 级字段
    if pack_cols:
        pack_agg = (
            df[key_cols + pack_cols].drop_duplicates(subset=key_cols, keep="last").set_index(key_cols)
        )
        out = pack_agg.join(pivot, how="outer").reset_index()
    else:
        out = pivot.reset_index()

    # 追加 4 列：Tlow/Thigh/Vlow/Vhigh（基于转换后的列名）
    import numpy as np

    tempr_cols = [c for c in out.columns if c.startswith(f"{series_prefix}Tempr_")]
    volt_cols = [c for c in out.columns if c.startswith(f"{series_prefix}Volt_")]

    if tempr_cols:
        tvals = out[tempr_cols].apply(pd.to_numeric, errors="coerce")
        out["Tlow"] = tvals.min(axis=1, skipna=True)
        out["Thigh"] = tvals.max(axis=1, skipna=True)
    else:
        out["Tlow"] = np.nan
        out["Thigh"] = np.nan

    if volt_cols:
        vvals = out[volt_cols].apply(pd.to_numeric, errors="coerce")
        out["Vlow"] = vvals.min(axis=1, skipna=True)
        out["Vhigh"] = vvals.max(axis=1, skipna=True)
    else:
        out["Vlow"] = np.nan
        out["Vhigh"] = np.nan

    # 列顺序：meta/time 在前 → pack → 电芯展开列 → 4 个聚合
    lead = [c for c in ("Txt_File_Name", "txt_file_name", "Data_Source", "data_source", "Time") if c in out.columns]
    agg4 = ["Tlow", "Thigh", "Vlow", "Vhigh"]
    batwide = [c for c in out.columns if c.startswith(series_prefix)]
    others = [c for c in out.columns if c not in lead + batwide + agg4]
    out = out[lead + others + batwide + agg4]

    # 归并 BaseState（只保留第一列，基于转换后的1-15）
    base_state_cols = [
        col
        for col in out.columns
        if re.match(r"^Bat_BaseState_\d+$", col) and int(col.split("_")[-1]) in range(1, max_cells + 1)  # 适配1-15
    ]
    if base_state_cols:
        out = out.rename(columns={base_state_cols[0]: "BaseState"})
        out = out.drop(columns=base_state_cols[1:])

    if "bat_events" in df.columns:
        out = out.rename(columns={"bat_events": "Events"})

    return out


# =========================
# 解析：System 表（列中包含 time 但非完整时间戳）
# =========================

def parse_system_table(lines: List[str], header_idx: int) -> Optional[pd.DataFrame]:
    header = lines[header_idx]
    columns = re.sub(r"base\s*state", "base_state", header, flags=re.IGNORECASE)
    columns = re.sub(r"\.\s*", "_", columns.lower()).strip().split()

    info_system: Dict[str, List[Optional[str]]] = {}
    real_col_len = len(columns)
    for n_col, col in enumerate(columns):
        if any(x in col for x in ("err", "Err", "ERR", "event", "Event", "EVENT")):
            real_col_len = n_col
            break
        else:
            info_system[col] = []

    info_system.setdefault("error_code", [])
    info_system.setdefault("events", [])

    for j in range(header_idx + 1, len(lines)):
        row = lines[j]
        if CMD_COMPLETED_RE.search(row) or CMD_FAIL_RE.search(row):
            break
        row_list_original = row.strip().split()
        if len(row_list_original) < max(0, len(columns) - 6) or ("Press" in row_list_original and "exit" in row_list_original):
            continue
        row_list: List[str] = []
        for n, element in enumerate(row_list_original):
            if element.count("-") == 2:
                continue
            if element.count(":") == 2 or element.count("：") == 2:
                row_list.append(row_list_original[n - 1] + " " + element)
                continue
            row_list.append(element)

        add_error_code = False
        add_events = False
        for p, q in enumerate(row_list):
            if p < real_col_len:
                col_name = columns[p]
                info_system.setdefault(col_name, []).append(q)
            else:
                if "0x" in q and "0x" not in row_list[p - 1]:
                    if p == len(row_list) - 1:
                        info_system["error_code"].append(q)
                        add_error_code = True
                    else:
                        for r in range(p + 1, len(row_list)):
                            if "0x" not in row_list[r]:
                                info_system["error_code"].append(" ".join(row_list[p:r]))
                                add_error_code = True
                                break
                if "0x" not in q and p == real_col_len:
                    if p == len(row_list) - 1:
                        info_system["events"].append(q)
                        add_events = True
                    else:
                        for r in range(p + 1, len(row_list)):
                            if "0x" in row_list[r]:
                                info_system["events"].append(" ".join(row_list[p:r]))
                                add_events = True
                                break
                            if "0x" not in row_list[r] and r == len(row_list) - 1:
                                info_system["events"].append(" ".join(row_list[p : r + 1]))
                                add_events = True
                if "0x" not in q and "0x" in row_list[p - 1]:
                    info_system["events"].append(" ".join(row_list[p:]))
                    add_events = True
        if not add_error_code:
            info_system["error_code"].append(None)
        if not add_events:
            info_system["events"].append(None)
        

    return pd.DataFrame.from_dict(info_system, orient="columns") if info_system else None

# =========================
# 解析：Battery 长列（列数 > 50，含重复 id/soc 等）
# =========================

def parse_battery_longcolumns_table(lines: List[str], header_idx: int) -> Optional[pd.DataFrame]:
    header = lines[header_idx]
    columns = re.sub(r"base\s*state", "base_state", header, flags=re.IGNORECASE)
    columns = re.sub(r"\.\s*", "_", columns.lower()).strip().split()

    # 1) 建立列名容器并确定 bat_start_index（第二个 id 的位置）
    seen = set()
    bat_start_index = None
    for n_col, col in enumerate(columns):
        if col in seen:
            if col == "id" and bat_start_index is None:
                bat_start_index = n_col
                break
        else:
            seen.add(col)
    if bat_start_index is None:
        return None

    def _map_repeat_col(c: str) -> str:
        if c == "id":
            return "b_id"
        if c == "soc":
            return "bsoc"
        return c

    rows_out: List[Dict[str, str]] = []

    for j in range(header_idx + 1, len(lines)):
        raw = lines[j]
        if CMD_COMPLETED_RE.search(raw):
            break

        row = raw.strip().split()
        if len(row) < 50:  # 经验阈值
            continue

        # 3.1 先收集“基底列”（bat_start_index 之前的一次性字段）
        base_record: Dict[str, str] = {}
        for r in range(0, min(bat_start_index, len(columns))):
            col = columns[r]
            base_record[col] = row[r] if r < len(row) else ""

        # 3.2 从 bat_start_index 起，把行切成多个 “id…(到下一个id前)” 的分段
        q = bat_start_index
        while q < len(columns):
            if columns[q] != "id":
                q += 1
                continue

            next_id = None
            for t in range(q + 1, len(columns)):
                if columns[t] == "id":
                    next_id = t
                    break
            seg_end = next_id if next_id is not None else len(columns)

            rec = dict(base_record)
            for r in range(q, seg_end):
                if r >= len(row):
                    break
                col_mapped = _map_repeat_col(columns[r])
                rec[col_mapped] = row[r]

            rows_out.append(rec)
            if next_id is None:
                break
            q = next_id

    return pd.DataFrame(rows_out) if rows_out else None


# =========================
# Battery “长表→宽表”
# =========================

def battery_long_to_wide(
    df: pd.DataFrame,
    *,
    max_cells: int = 16,
    series_prefix: str = "Bat_",
    fill_value: str = "",
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()

    # 1) 合并 day+time -> time（把 time 放回原位置）
    if "day" in df.columns and "time" in df.columns:
        time_idx = list(df.columns).index("time")
        df["time"] = df["day"].astype(str) + " " + df["time"].astype(str)
        df = df.drop(columns=["day"])
        cols = list(df.columns)
        cols.remove("time")
        cols.insert(time_idx, "time")
        df = df[cols]

    # 2) 电芯编号列：优先 b_id，否则 id
    id_col = "b_id" if "b_id" in df.columns else ("id" if "id" in df.columns else None)
    if id_col is None:
        raise ValueError("未找到电芯编号列（期望 b_id 或 id）。")

    # 3) 电芯级字段：以 'b' 开头但不是编号列
    battery_cols = [c for c in df.columns if c.startswith("b") and c != id_col]
    for c in ("bvolt", "btempr", "bvolt_st", "btemp_st", "bsoc", "bbase_st", "bbase_state"):
        if c in df.columns and c not in battery_cols and c != id_col:
            battery_cols.append(c)
    if not battery_cols:
        key_cols = [c for c in ("Txt_File_Name", "txt_file_name", "Data_Source", "data_source", "time") if c in df.columns]
        return df.drop_duplicates(subset=key_cols)

    # 4) 键列：每行 = 一个时间点
    key_cols = [c for c in ("Txt_File_Name", "txt_file_name", "Data_Source", "data_source", "time") if c in df.columns]
    if "time" not in key_cols:
        raise ValueError("未检测到 time 列（需要先合并 day+time 或确保已有 time 列）。")

    def _to_int_safe(x):
        try:
            return int(str(x).strip())
        except Exception:
            return None

    # 5) 透视前瘦身并去重
    slim = df[key_cols + [id_col] + battery_cols].copy()
    slim = (
        slim.sort_values(key_cols + [id_col]).drop_duplicates(subset=key_cols + [id_col], keep="last")
    )

    pivot = slim.pivot_table(index=key_cols, columns=id_col, values=battery_cols, aggfunc="first")

    # 6) 指标命名映射
    def _metric_base(metric: str) -> str:
        return metric[1:] if metric.startswith("b") else metric

    SPECIAL_MAP = {
        "soc": "SOC",
        "tempr": "Tempr",
        "volt": "Volt",
        "volt_st": "State",
        "temp_st": "Temp_State",
        "base_st": "BaseState",
        "base_state": "BaseState",
        "ase_st": "BaseState",
        "time": "Time",
    }

    def _metric_pretty(metric: str) -> str:
        base = _metric_base(metric)
        key = base.lower()
        if key in SPECIAL_MAP:
            return SPECIAL_MAP[key]
        return base[:1].upper() + base[1:]

    col_pairs = []
    for metric, bid in pivot.columns:
        pretty = _metric_pretty(metric)
        new_name = f"{series_prefix}{pretty}_{bid}"
        sort_key = _to_int_safe(bid) if _to_int_safe(bid) is not None else 10**9
        col_pairs.append((sort_key, new_name, (metric, bid)))

    col_pairs.sort(key=lambda x: x[0])
    ordered_cols = [cp[2] for cp in col_pairs]
    new_names = [cp[1] for cp in col_pairs]
    pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(ordered_cols))
    pivot.columns = new_names

    metrics_set = sorted(set(battery_cols))
    target_cols = []
    for m in metrics_set:
        pretty = _metric_pretty(m)
        for i in range(1, max_cells + 1):
            target_cols.append(f"{series_prefix}{pretty}_{i}")

    current_cols = set(pivot.columns)
    missing_cols = [c for c in target_cols if c not in current_cols]
    if missing_cols:
        for c in missing_cols:
            pivot[c] = fill_value
    pivot = pivot.reindex(columns=target_cols)

    exclude = set(key_cols + [id_col] + battery_cols)
    pack_cols = [c for c in df.columns if c not in exclude]
    if pack_cols:
        pack_agg = (
            df[key_cols + pack_cols].drop_duplicates(subset=key_cols, keep="last").set_index(key_cols)
        )
        out = pack_agg.join(pivot, how="outer").reset_index()
    else:
        out = pivot.reset_index()

    lead = [c for c in ("Txt_File_Name", "txt_file_name", "Data_Source", "data_source", "Time") if c in out.columns]
    others = [c for c in out.columns if c not in lead and not c.startswith(series_prefix)]
    batwide = [c for c in out.columns if c.startswith(series_prefix)]
    out = out[lead + others + batwide]

    base_state_cols = [
        col
        for col in out.columns
        if re.match(r"^Bat_BaseState_\d+$", col) and int(col.split("_")[-1]) in range(1, 17)
    ]
    if base_state_cols:
        out = out.rename(columns={base_state_cols[0]: "BaseState"})
        out = out.drop(columns=base_state_cols[1:])

    if "Events" not in out.columns:
        out["Events"] = ""

    return out

# =========================
# 单文件解析（组合上述解析器）
# =========================

def parse_single_file(txt_path: str) -> Dict[str, List[pd.DataFrame]]:
    lines = read_text_lines(txt_path)

    # 统一的 meta & info/stat
    meta = parse_general_meta(txt_path, lines)
    info_dict, stat_dict = parse_info_stat_sections(lines)

    df_battery_list: List[pd.DataFrame] = []
    df_system_list: List[pd.DataFrame] = []
    df_battery_longcolumns_list: List[pd.DataFrame] = []
    df_info_stat_rows: List[pd.DataFrame] = []

    for idx, line in enumerate(lines):
        # 1) 基于完整时间戳的 battery 短表触发行
        if (
            re.search(r"\bTime\b", line, re.IGNORECASE)
            and not re.search(r"\bdevice\b", line, re.IGNORECASE)
            and TIMESTAMP_RE.search(line)
        ):
            df_bat = parse_battery_short_blocks(lines, idx)
            if df_bat is not None and not df_bat.empty:
                df_out = df_bat.copy()
                for k in ("Txt_File_Name", "Data_Source"):
                    df_out[k] = meta.get(k)
                lead_cols = [c for c in ("Txt_File_Name", "Data_Source") if c in df_out.columns]
                other_cols = [c for c in df_out.columns if c not in lead_cols]
                df_out = df_out[lead_cols + other_cols]
                df_battery_list.append(df_out)

        # 2) 列标题包含 time（但非完整时间戳行）
        if TIME_WORD_RE.search(line) and not TIMESTAMP_RE.search(line):
            columns = re.sub(r"base\s*state", "base_state", line, flags=re.IGNORECASE)
            columns = re.sub(r"\.\s*", "_", columns.lower()).strip().split()

            if len(columns) <= 50:
                df_sys = parse_system_table(lines, idx)
                if df_sys is not None and not df_sys.empty:
                    df_sys_out = df_sys.copy()
                    for k in ("Txt_File_Name", "Data_Source"):
                        df_sys_out[k] = meta.get(k)
                    lead_cols = [c for c in ("Txt_File_Name", "Data_Source") if c in df_sys_out.columns]
                    other_cols = [c for c in df_sys_out.columns if c not in lead_cols]
                    df_sys_out = df_sys_out[lead_cols + other_cols]
                    df_system_list.append(df_sys_out)
            else:
                df_long = parse_battery_longcolumns_table(lines, idx)
                if df_long is not None and not df_long.empty:
                    df_long_out = df_long.copy()
                    for k in ("Txt_File_Name", "Data_Source"):
                        df_long_out[k] = meta.get(k)
                    lead_cols = [c for c in ("Txt_File_Name", "Data_Source") if c in df_long_out.columns]
                    other_cols = [c for c in df_long_out.columns if c not in lead_cols]
                    df_long_out = df_long_out[lead_cols + other_cols]
                    df_battery_longcolumns_list.append(df_long_out)

    # 生成“每文件一行”的 info/stat 行表
    df_info_stat_rows.append(
        pd.DataFrame(
            [
                {
                    "Txt_File_Name": meta.get("Txt_File_Name"),
                    "Data_Source": meta.get("Data_Source"),
                    "info": info_dict,
                    "stat": stat_dict,
                }
            ]
        )
    )

    return {
        "df_system": df_system_list,
        "df_battery": df_battery_list,
        "df_battery_longcolumns": df_battery_longcolumns_list,
        "df_info_stat": df_info_stat_rows,
    }

# =========================
# 批量文件处理
# =========================

def _extract_numeric_series(s: pd.Series) -> pd.Series:
    if s is None:
        return s
    mask = s.notna()
    s2 = s.copy()
    s2.loc[mask] = s.loc[mask].astype(str).str.extract(r"([-+]?\d+\.?\d*)")[0]
    return s2


def clean_system_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "release_date" in df.columns:
        mask = df["release_date"].notna()
        if mask.any():
            df.loc[mask, "release_date"] = df.loc[mask, "release_date"].apply(year_standardization)
    if "time" in df.columns:
        mask = df["time"].notna()
        if mask.any():
            df.loc[mask, "time"] = df.loc[mask, "time"].apply(year_standardization)
        df = df[df["time"].notna()]
        df = df.rename(columns={"time": "Time"})
    rename_num = {
        "max_dischg_curr": "max_dischg_curr_ma",
        "max_charge_curr": "max_charge_curr_ma",
    }
    for raw, new in rename_num.items():
        if raw in df.columns:
            df.loc[:, raw] = _extract_numeric_series(df[raw])
            df = df.rename(columns={raw: new})
    mapping = {
        "per%": "SOC",
        "vo(mv)": "Volt",
        "cu(ma)": "Curr",
        "tempr": "Tempr",
        "tlow": "Tlow",
        "thigh": "Thigh",
        "vlowest": "Vlow",
        "vhighest": "Vhigh",
        "base_st": "BaseState",
        "volt_st": "Volt_State",
        "curr_st": "Curr_State",
        "temp_st": "Temp_State",
        "error_code": "ErrCode",
        "events": "Events",
        "item": "Item",
    }
    df['per%'] = df['per%'].str.replace('%', '').astype(int)

    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def clean_battery_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "release_date" in df.columns:
        mask = df["release_date"].notna()
        if mask.any():
            df.loc[mask, "release_date"] = df.loc[mask, "release_date"].apply(year_standardization)
    if "time" in df.columns:
        mask = df["time"].notna()
        if mask.any():
            df.loc[mask, "time"] = df.loc[mask, "time"].apply(year_standardization)
        df = df[df["time"].notna()]
        df = df.rename(columns={"time": "Time"})

    numeric_map = {
        "max_dischg_curr": "max_dischg_curr_ma",
        "max_charge_curr": "max_charge_curr_ma",
        "voltage": "voltage_mv",
        "current": "current_ma",
        "temperature": "temperature_mc",
        "total_coulomb": "total_coulomb_mah",
        "max_voltage": "max_voltage_mv",
        "percent": "percent",  # 先提数值，后统一重命名为 SOC
        "coulomb": "coulomb",
    }
    for raw, new in numeric_map.items():
        if raw in df.columns:
            df.loc[:, raw] = _extract_numeric_series(df[raw])
            if raw != new:
                df = df.rename(columns={raw: new})

    mapping = {
        "voltage_mv": "Volt",
        "current_ma": "Curr",
        "temperature_mc": "Tempr",
        "percent": "SOC",
        "total_coulomb_mah": "TotalCoul",
        "max_voltage_mv": "Max_Voltage",
        "Bat Events": "Events",
        "volt_state": "Volt_State",
        "curr_state": "Curr_State",
        "temp_state": "Temp_State",
        "item_index": "Item",
        "power_events": "ErrCode",
    }
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def clean_battery_long_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "release_date" in df.columns:
        mask = df["release_date"].notna()
        if mask.any():
            df.loc[mask, "release_date"] = df.loc[mask, "release_date"].apply(year_standardization)
    if "day" in df.columns:
        mask = df["day"].notna()
        if mask.any():
            df.loc[mask, "day"] = df.loc[mask, "day"].apply(year_standardization)

    for col in ("max_dischg_curr", "max_charge_curr", "soc", "bsoc"):
        if col in df.columns:
            df.loc[:, col] = _extract_numeric_series(df[col])

    rename_map = {
        "max_dischg_curr": "max_dischg_curr_ma",
        "max_charge_curr": "max_charge_curr_ma",
        "id": "Item",
        "volt": "Volt",
        "curr": "Curr",
        "tempr": "Tempr",
        "tlow": "Tlow",
        "thigh": "Thigh",
        "vlowest": "Vlow",
        "vhighest": "Vhigh",
        "base_st": "State",
        "volt_st": "Volt_State",
        "curr_st": "Curr_State",
        "temp_st": "Temp_State",
        "error_code": "ErrCode",
        "events": "Events",
        "soc": "SOC",
        "errcode": "ErrCode",
        "most_st": "Most_State",
        "totalcoul": "TotalCoul",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# =========================
# 合并结果 & 列名收集
# =========================

def collect_schema(frames: List[pd.DataFrame]) -> List[str]:
    schema: List[str] = []
    seen = set()
    for df in frames:
        for col in df.columns:
            if col not in seen:
                seen.add(col)
                schema.append(col)
    return schema


def merge_category_frames(df_lists: Dict[str, List[pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
    merged: Dict[str, pd.DataFrame] = {}
    for k, frames in df_lists.items():
        if not frames:
            continue
        schema = collect_schema(frames)
        df_total = pd.DataFrame(columns=schema)
        for df in tqdm(frames, desc=f"merge {k}"):
            aligned = df.reindex(columns=schema)
            df_total = pd.concat([df_total, aligned], ignore_index=True)
        merged[k] = df_total
    return merged


# =========================
# 顶层：批量处理入口
# =========================

def process_files(txt_path_list: List[str]) -> Dict[str, List[pd.DataFrame]]:
    """逐文件解析并聚合四类列表；三类明细表仍走清洗流程；info/stat 行表保持原样（dict 列）。"""
    df_battery_list: List[pd.DataFrame] = []
    df_system_list: List[pd.DataFrame] = []
    df_battery_longcolumns_list: List[pd.DataFrame] = []
    df_info_stat_rows: List[pd.DataFrame] = []

    for txt_path in tqdm(txt_path_list, desc="parse"):
        parsed = parse_single_file(txt_path)
        df_battery_list.extend(parsed.get("df_battery", []))
        df_system_list.extend(parsed.get("df_system", []))
        df_battery_longcolumns_list.extend(parsed.get("df_battery_longcolumns", []))
        df_info_stat_rows.extend(parsed.get("df_info_stat", []))

    # 三类明细表走清洗
    df_system_list = [clean_system_df(df) for df in tqdm(df_system_list, desc="clean system")]
    df_battery_list = [clean_battery_df(df) for df in tqdm(df_battery_list, desc="clean battery")]
    df_battery_longcolumns_list = [clean_battery_long_df(df) for df in tqdm(df_battery_longcolumns_list, desc="clean battery")]
    # 延后长表清洗与宽表转换到 __main__，此处保持原始长表
    # df_battery_longcolumns_list 保持不变

    return {
        "df_system": df_system_list,
        "df_battery": df_battery_list,
        "df_battery_longcolumns": df_battery_longcolumns_list,
        "df_info_stat": df_info_stat_rows,
    }


# if __name__ == "__main__":
#     folder_path = os.environ.get("LOG_FOLDER", r"D:\steamlit_售后\log_process\long_cloums")
#     log.info(f"扫描目录: {folder_path}")

#     file_path_list = find_all_file_paths(folder_path)
#     log.info(f"发现 txt 文件: {len(file_path_list)}")

#     parsed_lists = process_files(file_path_list)

#     # 1) 合并三类明细
#     merged_dfs = merge_category_frames({k: v for k, v in parsed_lists.items() if k != "df_info_stat"})

#     df_battery_total = merged_dfs.get("df_battery")
#     df_system_total = merged_dfs.get("df_system")
#     df_battery_longcolumns_total = merged_dfs.get("df_battery_longcolumns")

#     # 2) battery 短表转宽
#     if df_battery_total is not None and not df_battery_total.empty:
#         df_battery_total_wide = battery_total_to_wide(
#             df_battery_total, max_cells=16, series_prefix="Bat_", fill_value=""
#         )
#         #df_battery_total_wide.to_csv("df_battery_total_wide.csv", index=False)
#         log.info(f"df_battery_total_wide 已保存，shape={df_battery_total_wide.shape}")
#     else:
#         log.warning("无 df_battery 数据，跳过宽表导出。")

#     if df_system_total is not None and not df_system_total.empty:
#         #df_system_total.to_csv("system_total.csv", index=False)
#         log.info(f"system_total.csv 已保存，shape={df_system_total.shape}")
#     else:
#         log.warning("无 df_system 数据，跳过导出。")

#     if df_battery_longcolumns_total is not None and not df_battery_longcolumns_total.empty:
#         # 先转宽表，再导出（与 df_battery_total 流程一致）
#         df_battery_longcolumns_total_wide = battery_long_to_wide(
#             df_battery_longcolumns_total, max_cells=16, series_prefix="Bat_", fill_value=""
#         )
#         # 统一列名：time -> Time（与短表保持一致）
#         if "time" in df_battery_longcolumns_total_wide.columns:
#             df_battery_longcolumns_total_wide = df_battery_longcolumns_total_wide.rename(columns={"time": "Time"})
#         #df_battery_longcolumns_total_wide.to_csv("battery_longcolumns_total_wide.csv", index=False)
#         log.info(
#             f"battery_longcolumns_total_wide 已保存，shape={df_battery_longcolumns_total_wide.shape}"
#         )
#     else:
#         log.warning("无 battery_longcolumns 数据，跳过宽表导出。")

#     # 3) 合并 info/stat（一个文件一行）
#     info_stat_frames = parsed_lists.get("df_info_stat", [])
#     if info_stat_frames:
#         df_info_stat_total = pd.concat(info_stat_frames, ignore_index=True)
#         # df_info_stat_total.to_csv("info_stat_total.csv", index=False)
#         df_info_stat_total.to_json("info_stat_total.json", orient="records", force_ascii=False)
#         log.info(
#             f"info_stat_total.[csv|json] 已保存，shape={df_info_stat_total.shape}"
#         )
#     else:
#         log.warning("无 info/stat 数据，跳过导出。")

#     common_columns_battery = df_battery_longcolumns_total_wide.columns.intersection(
#         df_battery_total_wide.columns
#     ).tolist()
#     print("电池数据公共列：", common_columns_battery)

#     # 筛选公共列（使用copy()避免视图修改警告）
#     df1_common = df_battery_longcolumns_total_wide[common_columns_battery].copy()
#     df2_common = df_battery_total_wide[common_columns_battery].copy()

#     # 定义需要转换为数值类型的列（根据业务场景预设）
#     numeric_cols = ['Tlow', 'Thigh', 'Vlow', 'Vhigh']

#     # 转换公共列中的数值列类型为float（非数值转为NaN）
#     for col in numeric_cols:
#         if col in common_columns_battery:  # 仅处理存在于公共列中的数值列
#             df1_common[col] = pd.to_numeric(df1_common[col], errors='coerce')  # 左表转换
#             df2_common[col] = pd.to_numeric(df2_common[col], errors='coerce')  # 右表转换（补充原代码中遗漏的右表转换）

#     # --------------------------
#     # 步骤2：合并前两个DataFrame得到df_outer
#     # --------------------------
#     # 基于公共列外连接合并（保留所有行和公共列）
#     df_outer = pd.merge(
#         df1_common,
#         df2_common,
#         on=common_columns_battery,  # 公共列作为合并键
#         how='outer',                # 外连接：保留所有行
#         suffixes=('_left', '_right')# 重名列后缀区分（若有）
#     )
#     print("前两个DataFrame合并后形状：", df_outer.shape)

#     # --------------------------
#     # 步骤3：处理df_outer与系统数据的合并
#     # --------------------------
#     # 转换系统数据中的数值列类型（确保与df_outer类型一致）
#     for col in numeric_cols:
#         if col in df_system_total.columns:  # 仅处理系统数据中存在的数值列
#             df_system_total[col] = pd.to_numeric(df_system_total[col], errors='coerce')

#     # 提取df_outer与系统数据的公共列
#     common_cols_final = df_outer.columns.intersection(df_system_total.columns).tolist()
#     print("最终合并公共列：", common_cols_final)

#     # 外连接合并df_outer与系统数据
#     merged_df = pd.merge(
#         df_outer,
#         df_system_total,
#         on=common_cols_final,  # 最终公共列作为合并键
#         how='outer',           # 外连接：保留所有行和列
#         suffixes=('', '_dup')  # 重复非公共列后缀区分（保留原始列）
#     )

#     # 移除可能的重复列（带_dup后缀的列）
#     merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_dup')]
#     merged_df.to_csv("df_battery.csv")
#     print("最终合并结果形状：", merged_df.shape)

def process_logs(folder_path: str):


    """
    处理日志文件并返回合并后的信息统计DataFrame和最终合并数据
    
    参数:
        folder_path: 日志文件所在目录路径
        
    返回:
        df_info_stat_total: 信息统计合并DataFrame（若无可为空DataFrame）
        merged_df: 电池数据与系统数据的最终合并DataFrame（若无可为空DataFrame）
    """
    try:
        log.info(f"扫描目录: {folder_path}")

        # 1. 扫描目录获取文件路径
        file_path_list = find_all_file_paths(folder_path)  # 假设该函数已定义
        log.info(f"发现 txt 文件: {len(file_path_list)}")

        # 2. 处理文件得到解析结果
        parsed_lists = process_files(file_path_list)  # 假设该函数已定义

        # 3. 合并三类明细数据
        merged_dfs = merge_category_frames({
            k: v for k, v in parsed_lists.items() if k != "df_info_stat"
        })  # 假设该函数已定义

        # 提取各类数据（处理空数据情况）
        df_battery_total = merged_dfs.get("df_battery") if merged_dfs else None
        df_system_total = merged_dfs.get("df_system") if merged_dfs else None
        df_battery_longcolumns_total = merged_dfs.get("df_battery_longcolumns") if merged_dfs else None

        # 4. 处理电池短表转宽表
        df_battery_total_wide = None
        if df_battery_total is not None and not df_battery_total.empty:
            df_battery_total_wide = battery_total_to_wide(  # 假设该函数已定义
                df_battery_total, max_cells=16, series_prefix="Bat_", fill_value=""
            )
            
            log.info(f"df_battery_total_wide 生成，shape={df_battery_total_wide.shape}")
        else:
            log.warning("无 df_battery 数据，跳过短表转宽表。")

        # 5. 处理电池长表转宽表
        df_battery_longcolumns_total_wide = None
        if df_battery_longcolumns_total is not None and not df_battery_longcolumns_total.empty:
            df_battery_longcolumns_total_wide = battery_long_to_wide(  # 假设该函数已定义
                df_battery_longcolumns_total, max_cells=16, series_prefix="Bat_", fill_value=""
            )
            # 统一列名（time -> Time）
            if "time" in df_battery_longcolumns_total_wide.columns:
                df_battery_longcolumns_total_wide = df_battery_longcolumns_total_wide.rename(
                    columns={"time": "Time"}
                )
            log.info(f"battery_longcolumns_total_wide 生成，shape={df_battery_longcolumns_total_wide.shape}")
        else:
            log.warning("无 battery_longcolumns 数据，跳过长表转宽表。")

        # 6. 处理 info/stat 数据合并
        df_info_stat_total = pd.DataFrame()  # 默认为空DataFrame
        info_stat_frames = parsed_lists.get("df_info_stat", []) if parsed_lists else []
        if info_stat_frames:
            df_info_stat_total = pd.concat(info_stat_frames, ignore_index=True)
            log.info(f"info_stat_total 生成，shape={df_info_stat_total.shape}")
        else:
            log.warning("无 info/stat 数据，info_stat_total 为空。")

        # 7. 合并电池长表宽表与短表宽表
        df_outer = pd.DataFrame()  # 默认为空DataFrame
        if df_battery_longcolumns_total_wide is not None and df_battery_total_wide is not None:
            # 提取公共列
            common_columns_battery = df_battery_longcolumns_total_wide.columns.intersection(
                df_battery_total_wide.columns
            ).tolist()
            log.info(f"电池数据公共列: {common_columns_battery}")

            # 筛选公共列并复制（避免视图警告）
            df1_common = df_battery_longcolumns_total_wide[common_columns_battery].copy()
            df2_common = df_battery_total_wide[common_columns_battery].copy()

            # 数值列类型转换（确保合并兼容性）
            numeric_cols = ['Tlow', 'Thigh', 'Vlow', 'Vhigh']
            for col in numeric_cols:
                if col in common_columns_battery:
                    df1_common[col] = pd.to_numeric(df1_common[col], errors='coerce')
                    df2_common[col] = pd.to_numeric(df2_common[col], errors='coerce')

            # 外连接合并
            df_outer = pd.merge(
                df1_common,
                df2_common,
                on=common_columns_battery,
                how='outer',
                suffixes=('_left', '_right')
            )
            log.info(f"电池长表与短表合并后形状: {df_outer.shape}")
        else:
            log.warning("电池长表或短表宽表为空，跳过电池数据合并。")

        # 8. 合并电池合并数据与系统数据
        merged_df = pd.DataFrame()  # 默认为空DataFrame
        numeric_cols = ['Volt','Curr','Tempr','Tlow', 'Thigh', 'Vlow', 'Vhigh','SOC','Bat_Volt_1','Bat_Volt_2','Bat_Volt_3','Bat_Volt_4','Bat_Volt_5',
                        'Bat_Volt_6','Bat_Volt_7','Bat_Volt_8','Bat_Volt_9','Bat_Volt_10','Bat_Volt_11','Bat_Volt_12','Bat_Volt_13',
                        'Bat_Volt_14','Bat_Volt_15','Bat_Volt_16',]
        
        # 对 df_outer 执行同样检查
        for col in numeric_cols:
            if col in df_outer.columns and df_outer[col].dtype == 'object':
                df_outer[col] = pd.to_numeric(df_outer[col], errors='coerce').astype('Int64')
                print(f"合并后修复 {col} 类型为 {df_outer[col].dtype}")
        
        for col in numeric_cols:
            if col in df_system_total.columns and df_system_total[col].dtype == 'object':
                df_system_total[col] = pd.to_numeric(df_system_total[col], errors='coerce').astype('Int64')
                print(f"合并后修复 {col} 类型为 {df_system_total[col].dtype}")
        
        merged_df = pd.concat([df_system_total, df_outer], ignore_index=True)

        #     # 外连接合并
        #     merged_df = pd.merge(
        #         df_outer,
        #         df_system_total,
        #         on=common_cols_final,
        #         how='outer',
        #         suffixes=('', '_dup')
        #     )

        #     # 移除重复列（带_dup后缀）
        #     merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_dup')]
        #     log.info(f"最终合并结果形状: {merged_df.shape}")
        # else:
        #     log.warning("电池合并数据或系统数据为空，最终合并结果为空。")
        return df_info_stat_total,merged_df
        #return df_info_stat_total, merged_df

    except Exception as e:
        log.error(f"处理过程出错: {str(e)}", exc_info=True)
        # 出错时返回空DataFrame
        return pd.DataFrame(), pd.DataFrame()
    

# 使用示例（需确保依赖函数已定义）:
if __name__ == "__main__":
    folder_path = os.environ.get("LOG_FOLDER", r"D:\steamlit_售后\log_process\long_cloums")
    #df_info_stat, merged_data = process_logs(folder_path)
    df_info_stat , merged_df = process_logs(folder_path)
    #print(merged_data)
#    # 1. 定义用于判断“行相同”的列（分组列）
#     group_cols = ['Txt_File_Name', 'Data_Source', 'Time', 'Item', 'Volt', 'Curr', 
#                   'Tempr', 'Tlow', 'Thigh', 'Vlow', 'Vhigh', 'Volt_State', 'Curr_State', 'Temp_State']
    

#     # 2. 自定义拼接函数：将组内元素转为字符串，用空格连接（跳过空值）
#     def join_strings(series):
#         # 处理空值（将NaN转为空字符串）
#         str_list = [str(x).strip() if pd.notna(x) else '' for x in series]
#         # 过滤空字符串，并用空格连接
#         return ' '.join([s for s in str_list if s])
#     # 3. 分组并应用拼接函数
#     merged_df = merged_data.groupby(group_cols, as_index=False).agg(join_strings)

#     print(f"info_stat 形状: {df_info_stat.shape}")
#     print(f"合并数据形状: {merged_data.shape}") 
