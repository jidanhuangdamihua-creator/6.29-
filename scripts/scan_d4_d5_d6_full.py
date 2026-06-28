"""
Full scan of Dataset4 (叮咚买菜), Dataset5 (Favorita), Dataset6 (M5)
Uses chunked pandas / pyarrow for large files.
"""
import os, sys, warnings, subprocess
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "数据集" / "原始数据"
D4   = BASE / "Dataset 4叮咚数据集"
D5   = BASE / "Dataset 5Favorita"
D6   = BASE / "Dataset 6m5-forecasting-accuracy"
SEP  = "=" * 70

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def fmt(x, d=4):
    if x is None: return "N/A"
    if isinstance(x, float): return f"{x:.{d}f}"
    return str(x)

def wc(path):
    r = subprocess.run(['wc', '-l', str(path)], capture_output=True, text=True)
    return r.stdout.strip().split()[0]

def df_to_md(df, max_rows=5):
    """Simple markdown table without tabulate dependency issues."""
    df = df.head(max_rows)
    cols = list(df.columns)
    rows = [cols, ["---"]*len(cols)]
    for _, row in df.iterrows():
        rows.append([str(v)[:40] for v in row.values])
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)

def print_col_summary(df, title=""):
    if title: print(f"\n**{title}**\n")
    print("| 列名 | dtype | n_unique | null_rate | 数值: min/max/mean  或  类别: 样本值 |")
    print("|------|-------|----------|-----------|-------------------------------------|")
    for col in df.columns:
        s = df[col]
        n = len(s)
        null_r = s.isna().sum() / n if n else 0
        if pd.api.types.is_numeric_dtype(s):
            nn = s.dropna()
            info = f"min={fmt(float(nn.min()) if len(nn) else None)}, max={fmt(float(nn.max()) if len(nn) else None)}, mean={fmt(float(nn.mean()) if len(nn) else None)}"
        else:
            try:
                uniq = s.nunique()
                vals = s.dropna().unique()[:10].tolist()
                info = f"{uniq} unique | {str(vals)[:70]}"
            except:
                info = "object/list"
        print(f"| {col} | {s.dtype} | {'?' if pd.api.types.is_object_dtype(s) else s.nunique()} | {null_r:.4f} | {info} |")

def accumulate_sales_stats(s):
    """Given a clean (dropna'd, numeric) Series, return stat dict."""
    arr = s.astype(float).values
    return dict(
        n=len(arr),
        mn=float(arr.min()),
        mx=float(arr.max()),
        sm=float(arr.sum()),
        sq=float((arr**2).sum()),
        zr=int((arr == 0).sum()),
        nr=int((arr < 0).sum()),
        sample=arr[np.random.choice(len(arr), min(len(arr), 1000), replace=False)].tolist() if len(arr) else [],
    )

def merge_stats(acc, new):
    acc['n']  += new['n']
    acc['mn']  = min(acc['mn'], new['mn']) if new['n'] else acc['mn']
    acc['mx']  = max(acc['mx'], new['mx']) if new['n'] else acc['mx']
    acc['sm'] += new['sm']
    acc['sq'] += new['sq']
    acc['zr'] += new['zr']
    acc['nr'] += new['nr']
    if len(acc['sample']) < 50000:
        acc['sample'].extend(new['sample'])
    return acc

def print_sales_stats(acc, null_count, total_rows, label=""):
    n = acc['n']
    if n == 0:
        print("  (无有效销售数据)")
        return
    mean = acc['sm'] / n
    var  = max(0, acc['sq']/n - mean**2)
    std  = np.sqrt(var)
    p_arr = np.array(sorted(acc['sample']))
    print(f"\n#### 销售值统计{'（'+label+'）' if label else ''}\n")
    print(f"| 指标 | 值 |")
    print(f"|------|----|")
    print(f"| 总非空行数 | {n:,} |")
    print(f"| min | {acc['mn']:.4f} |")
    print(f"| max | {acc['mx']:.4f} |")
    print(f"| mean | {mean:.4f} |")
    print(f"| std | {std:.4f} |")
    if len(p_arr):
        print(f"| median(approx) | {np.median(p_arr):.4f} |")
        print(f"| p1(approx) | {np.percentile(p_arr, 1):.4f} |")
        print(f"| p99(approx) | {np.percentile(p_arr, 99):.4f} |")
    print(f"| 零值率 | {acc['zr']/n:.4f} |")
    print(f"| 负值率 | {acc['nr']/n:.4f} |")
    print(f"| 缺失率 | {null_count/total_rows:.4f} |")

def print_entity_span_stats(spans, total_entities):
    if len(spans) == 0:
        print("  (无实体跨度数据)")
        return
    n210 = int((spans >= 210).sum())
    n365 = int((spans >= 365).sum())
    n730 = int((spans >= 730).sum())
    print(f"| 指标 | 值 |")
    print(f"|------|----|")
    print(f"| 实体总数 | {total_entities:,} |")
    print(f"| 最短历史跨度 | {int(spans.min())} 天 |")
    print(f"| 最长历史跨度 | {int(spans.max())} 天 |")
    print(f"| 平均历史跨度 | {spans.mean():.1f} 天 |")
    print(f"| ≥210天实体数 | {n210:,} ({n210/total_entities*100:.1f}%) |")
    print(f"| ≥365天实体数 | {n365:,} ({n365/total_entities*100:.1f}%) |")
    print(f"| ≥730天实体数 | {n730:,} ({n730/total_entities*100:.1f}%) |")
    return n210, n365, n730

# ──────────────────────────────────────────────
# DATASET 4 — 叮咚买菜
# ──────────────────────────────────────────────

def scan_d4():
    print(f"\n{SEP}")
    print("## Dataset 4 — 叮咚买菜 (Dingdong Fresh)")
    print(SEP)

    import pyarrow.parquet as pq
    train_pf = pq.ParquetFile(D4 / "data/train.parquet")
    eval_pf  = pq.ParquetFile(D4 / "data/eval.parquet")
    train_rows_total = train_pf.metadata.num_rows
    eval_rows_total  = eval_pf.metadata.num_rows
    n_groups         = train_pf.metadata.num_row_groups

    # ── File listing ──
    print("\n### 第一步：文件清单与角色识别\n")
    print("| 文件 | 大小 | 行数 | 角色 |")
    print("|------|------|------|------|")
    print(f"| data/train.parquet | 301 MB | {train_rows_total:,} | 主销售表 |")
    print(f"| data/eval.parquet  | 1.9 MB  | {eval_rows_total:,}  | 验证/评估销售表 |")
    print(f"| train_sample_100.csv | 21 KB | 100 | 主表样本(子集) |")
    print(f"| README.md | 4.7 KB | — | 数据说明文档 |")

    # ── Schema ──
    schema = train_pf.schema_arrow
    print(f"\n**Arrow Schema（训练集）：**\n```\n{schema}\n```")

    sample_chunk = train_pf.read_row_group(0).to_pandas()
    cols = list(sample_chunk.columns)

    # Identify scalar vs list columns
    list_cols   = [c for c in cols if sample_chunk[c].dtype == object
                   and isinstance(sample_chunk[c].iloc[0], (list, np.ndarray))]
    scalar_cols = [c for c in cols if c not in list_cols]
    print(f"\n**标量列（可直接统计）：** {scalar_cols}")
    print(f"**嵌套列表列（跳过直接统计）：** {list_cols}")

    # Define columns manually based on schema knowledge
    ENTITY_COLS = ['city_id', 'store_id', 'product_id']
    DATE_COL    = 'dt'
    SALES_COL   = 'sale_amount'
    FEATURE_COLS = [c for c in scalar_cols
                    if c not in ENTITY_COLS + [DATE_COL, SALES_COL]]

    print(f"\n**实体ID列：** {ENTITY_COLS}")
    print(f"**日期列：** {DATE_COL}")
    print(f"**目标列：** {SALES_COL}")
    print(f"**数值特征列：** {FEATURE_COLS}")

    print(f"\n**前5行（标量列）：**")
    print(df_to_md(sample_chunk[scalar_cols].head(5)))

    # ── Chunked stats: Pass 1 (sales + entity unique values) ──
    print(f"\n### 第二步：主表核心统计（chunked，逐row-group读取）\n")
    print("**进度：**")

    acc = dict(n=0, mn=np.inf, mx=-np.inf, sm=0.0, sq=0.0, zr=0, nr=0, sample=[])
    null_count   = 0
    total_rows   = 0
    date_min_g   = None
    date_max_g   = None
    all_dates    = set()
    entity_sets  = {c: set() for c in ENTITY_COLS}
    entity_date_min = {}
    entity_date_max = {}

    for i in range(n_groups):
        # Only read needed columns
        needed = ENTITY_COLS + [DATE_COL, SALES_COL]
        chunk = train_pf.read_row_group(i, columns=needed).to_pandas()
        total_rows += len(chunk)

        # Entity unique values
        for c in ENTITY_COLS:
            entity_sets[c].update(chunk[c].dropna().unique().tolist())

        # Dates
        dates = pd.to_datetime(chunk[DATE_COL], errors='coerce')
        chunk['_date'] = dates.dt.date
        mn, mx = dates.min(), dates.max()
        if date_min_g is None or mn < date_min_g: date_min_g = mn
        if date_max_g is None or mx > date_max_g: date_max_g = mx
        all_dates.update(chunk['_date'].dropna().unique().tolist())

        # Entity key for span tracking
        chunk['_eid'] = (chunk['city_id'].astype(str) + '_' +
                         chunk['store_id'].astype(str) + '_' +
                         chunk['product_id'].astype(str))
        grp = chunk.groupby('_eid')['_date'].agg(['min', 'max'])
        for eid, row in grp.iterrows():
            if eid not in entity_date_min or row['min'] < entity_date_min[eid]:
                entity_date_min[eid] = row['min']
            if eid not in entity_date_max or row['max'] > entity_date_max[eid]:
                entity_date_max[eid] = row['max']

        # Sales stats
        s = chunk[SALES_COL].dropna()
        null_count += int(chunk[SALES_COL].isna().sum())
        if len(s):
            new_acc = accumulate_sales_stats(s)
            acc = merge_stats(acc, new_acc)

        if (i+1) % 5 == 0 or i == n_groups - 1:
            print(f"  row_group {i+1}/{n_groups}, rows={total_rows:,}, entities={len(entity_date_min):,}")

    # Compute spans
    entity_spans_dict = {}
    for eid in entity_date_min:
        mn = entity_date_min[eid]
        mx = entity_date_max[eid]
        entity_spans_dict[eid] = (mx - mn).days + 1 if mn and mx else 0
    spans = np.array(list(entity_spans_dict.values()), dtype=float)
    total_entities = len(spans)

    print(f"\n#### 实体维度统计\n")
    print(f"| 指标 | 值 |")
    print(f"|------|----|")
    print(f"| 实体定义 | city_id × store_id × product_id |")
    for c in ENTITY_COLS:
        print(f"| {c} 唯一值数 | {len(entity_sets[c]):,} |")
    n210, n365, n730 = print_entity_span_stats(spans, total_entities)

    print_sales_stats(acc, null_count, total_rows, SALES_COL)

    print(f"\n#### 时间维度统计\n")
    all_dates_s = sorted(all_dates)
    print(f"| 指标 | 值 |")
    print(f"|------|----|")
    print(f"| 全局起始日期 | {date_min_g.date() if date_min_g else 'N/A'} |")
    print(f"| 全局结束日期 | {date_max_g.date() if date_max_g else 'N/A'} |")
    print(f"| 唯一日期数 | {len(all_dates_s)} |")
    if len(all_dates_s) > 1:
        gaps = [(all_dates_s[k+1]-all_dates_s[k]).days for k in range(len(all_dates_s)-1)]
        print(f"| 最大日期间隔 | {max(gaps)} 天 |")
        print(f"| gap>1天次数 | {sum(1 for g in gaps if g>1)} |")
        print(f"| 时间粒度 | 日 |")

    # ── Feature columns summary ──
    print(f"\n### 第三步：特征列摘要（train.parquet 标量非ID列）\n")
    feat_sample = train_pf.read_row_group(0, columns=FEATURE_COLS).to_pandas()
    print_col_summary(feat_sample, "特征列统计（样本 row_group 0）")
    print(f"\n**前3行：**")
    print(df_to_md(feat_sample.head(3)))

    # ── eval.parquet ──
    print(f"\n### eval.parquet 摘要\n")
    eval_df = eval_pf.read(columns=ENTITY_COLS + [DATE_COL, SALES_COL]).to_pandas()
    print(f"**行数：** {len(eval_df):,}")
    eval_dates = pd.to_datetime(eval_df[DATE_COL], errors='coerce')
    print(f"**日期范围：** {eval_dates.min().date()} ~ {eval_dates.max().date()}")
    print(f"**新实体（不在train中）：**", end=" ")
    eval_eids = set((eval_df['city_id'].astype(str)+'_'+eval_df['store_id'].astype(str)+'_'+eval_df['product_id'].astype(str)).unique())
    new_eids  = eval_eids - set(entity_date_min.keys())
    print(f"{len(new_eids):,} 个")

    # ── Join & RFE ──
    print(f"\n### 第四步：Join可行性评估\n")
    print("- 数据集仅含 train.parquet + eval.parquet，无外部辅助表")
    print("- eval 是同结构延续，直接追加即可，join key = city_id + store_id + product_id + dt")
    print(f"- RFE 可用特征列（排除实体ID和日期后）：{FEATURE_COLS + list_cols}")
    print(f"- 注意：hours_sale / hours_stock_status 是24小时列表，需展开或聚合统计量才能作为特征")

    print(f"\n### 第五步：冷启动可行性与 source/target 建议\n")
    print(f"- 推荐实体单位：city_id × store_id × product_id")
    print(f"- source pool（≥210天）：{n210:,} 个实体（{n210/total_entities*100:.1f}%）")
    print(f"- 冷启动 target 候选：eval 中的新实体 {len(new_eids):,} 个，或train中历史<210天的实体")
    print(f"- 纯数值可用 RFE 特征：{FEATURE_COLS}（共{len(FEATURE_COLS)}列）")
    print(f"- 高缺失列：需在全量扫描后确认 precpt/avg_temperature 等气象列的缺失率")

    return dict(
        dataset="D4 叮咚买菜",
        entity_def="city_id × store_id × product_id",
        total_entities=total_entities,
        date_range=f"{date_min_g.date() if date_min_g else '?'} ~ {date_max_g.date() if date_max_g else '?'}",
        unique_dates=len(all_dates_s),
        total_rows=total_rows,
        sales_col=SALES_COL,
        zero_rate=f"{acc['zr']/acc['n']:.4f}" if acc['n'] else "N/A",
        neg_rate=f"{acc['nr']/acc['n']:.4f}" if acc['n'] else "N/A",
        null_rate=f"{null_count/total_rows:.4f}",
        n_ge210=n210, pct_ge210=f"{n210/total_entities*100:.1f}%",
        n_ge365=n365, n_ge730=n730,
        aux_tables="eval.parquet（同结构延续）",
        granularity="日",
    )


# ──────────────────────────────────────────────
# DATASET 5 — Favorita
# ──────────────────────────────────────────────

def scan_d5():
    print(f"\n{SEP}")
    print("## Dataset 5 — Corporación Favorita (Kaggle 2018)")
    print(SEP)

    # ── File listing ──
    print("\n### 第一步：文件清单与角色识别\n")
    files_meta = [
        ("train.csv",             "4.7 GB",  "主销售表（store×item×date × unit_sales）"),
        ("test.csv",              "120 MB",  "预测目标期（无 unit_sales）"),
        ("stores.csv",            "1.4 KB",  "门店元数据（city, state, type, cluster）"),
        ("items.csv",             "99 KB",   "商品元数据（family, class, perishable）"),
        ("transactions.csv",      "1.5 MB",  "门店日级交易笔数辅助特征"),
        ("oil.csv",               "20 KB",   "外部特征：厄瓜多尔日原油价格"),
        ("holidays_events.csv",   "22 KB",   "外部特征：节假日（national/local）"),
        ("sample_submission.csv", "39 MB",   "提交样例（无特征价值）"),
    ]
    print("| 文件 | 大小 | 行数 | 角色 |")
    print("|------|------|------|------|")
    for fname, sz, role in files_meta:
        lines = wc(D5 / fname)
        print(f"| {fname} | {sz} | {lines} | {role} |")

    # ── Sample main table ──
    print("\n### 第二步：主表核心统计\n")
    sample = pd.read_csv(D5 / "train.csv", nrows=5)
    cols = list(sample.columns)
    print(f"**列名：** {cols}")
    print(f"\n**前5行：**")
    print(df_to_md(sample))

    ENTITY_COLS = ['store_nbr', 'item_nbr']
    DATE_COL    = 'date'
    SALES_COL   = 'unit_sales'
    PROMO_COL   = 'onpromotion' if 'onpromotion' in cols else None

    print(f"\n**chunked 扫描进度（chunksize=500,000）：**")
    CHUNKSIZE = 500_000
    acc       = dict(n=0, mn=np.inf, mx=-np.inf, sm=0.0, sq=0.0, zr=0, nr=0, sample=[])
    null_count = 0
    total_rows = 0
    date_min_g = None
    date_max_g = None
    all_dates  = set()
    store_set  = set()
    item_set   = set()
    entity_date_min = {}
    entity_date_max = {}
    chunk_i = 0

    for chunk in pd.read_csv(
            D5 / "train.csv", chunksize=CHUNKSIZE,
            dtype={'store_nbr': 'int16', 'item_nbr': 'int32',
                   'unit_sales': 'float32', 'onpromotion': 'object'}):
        chunk_i += 1
        total_rows += len(chunk)

        store_set.update(chunk['store_nbr'].unique().tolist())
        item_set.update(chunk['item_nbr'].unique().tolist())

        chunk['_eid'] = (chunk['store_nbr'].astype(str) + '_' +
                         chunk['item_nbr'].astype(str))

        dates = pd.to_datetime(chunk['date'], errors='coerce')
        chunk['_date'] = dates.dt.date
        mn, mx = dates.min(), dates.max()
        if date_min_g is None or mn < date_min_g: date_min_g = mn
        if date_max_g is None or mx > date_max_g: date_max_g = mx
        all_dates.update(chunk['_date'].dropna().unique().tolist())

        grp = chunk.groupby('_eid')['_date'].agg(['min', 'max'])
        for eid, row in grp.iterrows():
            if eid not in entity_date_min or row['min'] < entity_date_min[eid]:
                entity_date_min[eid] = row['min']
            if eid not in entity_date_max or row['max'] > entity_date_max[eid]:
                entity_date_max[eid] = row['max']

        s = chunk[SALES_COL].dropna()
        null_count += int(chunk[SALES_COL].isna().sum())
        if len(s):
            new_acc = accumulate_sales_stats(s)
            acc = merge_stats(acc, new_acc)

        if chunk_i % 20 == 0:
            print(f"  chunk {chunk_i}, rows={total_rows:,}, entities={len(entity_date_min):,}")

    print(f"  完成，共 {chunk_i} 个chunk，总行数={total_rows:,}")

    entity_spans_dict = {eid: (entity_date_max[eid]-entity_date_min[eid]).days+1
                         for eid in entity_date_min
                         if entity_date_min[eid] and entity_date_max[eid]}
    spans = np.array(list(entity_spans_dict.values()), dtype=float)
    total_entities = len(spans)

    print(f"\n#### 实体维度统计\n")
    print(f"| 指标 | 值 |")
    print(f"|------|----|")
    print(f"| 实体定义 | store_nbr × item_nbr |")
    print(f"| store 唯一值数 | {len(store_set):,} |")
    print(f"| item 唯一值数 | {len(item_set):,} |")
    print(f"| 实体总数(store×item) | {total_entities:,} |")
    n210, n365, n730 = print_entity_span_stats(spans, total_entities)

    print_sales_stats(acc, null_count, total_rows, SALES_COL)

    print(f"\n#### 时间维度统计\n")
    all_dates_s = sorted(all_dates)
    print(f"| 指标 | 值 |")
    print(f"|------|----|")
    print(f"| 全局起始日期 | {date_min_g.date() if date_min_g else 'N/A'} |")
    print(f"| 全局结束日期 | {date_max_g.date() if date_max_g else 'N/A'} |")
    print(f"| 唯一日期数 | {len(all_dates_s)} |")
    if len(all_dates_s) > 1:
        gaps = [(all_dates_s[k+1]-all_dates_s[k]).days for k in range(len(all_dates_s)-1)]
        print(f"| 最大日期间隔 | {max(gaps)} 天 |")
        print(f"| gap>1天次数 | {sum(1 for g in gaps if g>1)} |")
        print(f"| 时间粒度 | 日 |")

    # ── Auxiliary tables ──
    print("\n### 第三步：辅助表内容摘要\n")
    for aux in ['stores.csv', 'items.csv', 'transactions.csv', 'oil.csv', 'holidays_events.csv']:
        df = pd.read_csv(D5 / aux)
        print(f"\n#### {aux}  (行数: {len(df):,})\n")
        print_col_summary(df)
        print(f"\n**前3行：**")
        print(df_to_md(df, max_rows=3))

    # ── Join feasibility ──
    print("\n### 第四步：Join可行性评估\n")
    print("| 辅助表 | Join Key | 新增特征列 | 覆盖率估计 | 一对多风险 | Leakage风险 |")
    print("|--------|----------|------------|------------|------------|-------------|")
    print("| stores.csv | store_nbr | city,state,type,cluster | 100% | 无(1:1) | 无 |")
    print("| items.csv | item_nbr | family,class,perishable | 100% | 无(1:1) | 无 |")
    print("| transactions.csv | store_nbr × date | transactions | ~99%（部分日期缺失） | 无 | 无（同日粒度） |")
    print("| oil.csv | date | dcoilwtico | ~71%（仅工作日有数据，需插值） | 无 | 用前一日价格，当日值属于leakage |")
    print("| holidays_events.csv | date | locale,type,transferred | ~8%（节假日覆盖少，其余为0/无） | 可能(date → 多节日) | 无 |")

    print("\n### 第五步：冷启动可行性与RFE特征\n")
    rfe_features = ['store_type','store_cluster','store_city','store_state',
                    'item_family','item_class','item_perishable',
                    'onpromotion','transactions','dcoilwtico_lag1',
                    'is_holiday','holiday_type','holiday_locale']
    print(f"- **推荐实体单位：** store_nbr × item_nbr")
    print(f"- **source pool（≥210天）：** {n210:,} 个实体（{n210/total_entities*100:.1f}%）")
    print(f"- **冷启动target建议：** 某年后新上架商品，或人工截断末段作冷启动")
    print(f"- **RFE可用特征（join后，排除ID/日期/目标）：** {rfe_features}")
    print(f"- **需注意：** oil价格有工作日缺口，务必用前向填充(ffill)并lag 1天；holidays 需处理一对多关系（pivot最大/最小type）")

    return dict(
        dataset="D5 Favorita",
        entity_def="store_nbr × item_nbr",
        total_entities=total_entities,
        date_range=f"{date_min_g.date() if date_min_g else '?'} ~ {date_max_g.date() if date_max_g else '?'}",
        unique_dates=len(all_dates_s),
        total_rows=total_rows,
        sales_col=SALES_COL,
        zero_rate=f"{acc['zr']/acc['n']:.4f}" if acc['n'] else "N/A",
        neg_rate=f"{acc['nr']/acc['n']:.4f}" if acc['n'] else "N/A",
        null_rate=f"{null_count/total_rows:.4f}",
        n_ge210=n210, pct_ge210=f"{n210/total_entities*100:.1f}%",
        n_ge365=n365, n_ge730=n730,
        aux_tables="stores,items,transactions,oil,holidays_events",
        granularity="日",
    )


# ──────────────────────────────────────────────
# DATASET 6 — M5 Forecasting
# ──────────────────────────────────────────────

def scan_d6():
    print(f"\n{SEP}")
    print("## Dataset 6 — M5 Forecasting Accuracy (Walmart)")
    print(SEP)

    MAIN_FILE = D6 / "sales_train_evaluation.csv"

    # ── File listing ──
    print("\n### 第一步：文件清单与角色识别\n")
    files_meta = [
        ("sales_train_evaluation.csv", "116 MB", "主销售表（宽表，d_1~d_1941，共1941天）"),
        ("sales_train_validation.csv", "114 MB", "验证版主表（d_1~d_1913，共1913天）"),
        ("calendar.csv",               "101 KB", "时间特征表（date,weekday,month,event,snap）"),
        ("sell_prices.csv",            "194 MB", "价格表（store_id×item_id×wm_yr_wk → sell_price）"),
        ("sample_submission.csv",      "5.0 MB", "提交样例（无特征价值）"),
    ]
    print("| 文件 | 大小 | 行数 | 角色 |")
    print("|------|------|------|------|")
    for fname, sz, role in files_meta:
        lines = wc(D6 / fname)
        print(f"| {fname} | {sz} | {lines} | {role} |")

    # ── Schema ──
    print(f"\n### 第二步：主表结构分析（宽表格式）\n")
    header_df = pd.read_csv(MAIN_FILE, nrows=0)
    all_cols  = list(header_df.columns)
    ID_COLS   = [c for c in all_cols if not c.startswith('d_')]
    DAY_COLS  = [c for c in all_cols if c.startswith('d_')]

    print(f"**总列数：** {len(all_cols)}（{len(ID_COLS)} 个ID/元数据列 + {len(DAY_COLS)} 个销售天数列）")
    print(f"**ID/元数据列：** {ID_COLS}")
    print(f"**销售天数列范围：** d_1 ~ d_{len(DAY_COLS)}")

    # Row count
    wc_r = subprocess.run(['wc', '-l', str(MAIN_FILE)], capture_output=True, text=True)
    total_entities = int(wc_r.stdout.strip().split()[0]) - 1
    total_cells_equiv = total_entities * len(DAY_COLS)
    print(f"\n**宽表实体数（行数）：** {total_entities:,}")
    print(f"**等效长表总行数：** {total_cells_equiv:,}")

    # Sample
    sample = pd.read_csv(MAIN_FILE, nrows=5)
    print(f"\n**前5行（仅ID列 + d_1~d_5）：**")
    print(df_to_md(sample[ID_COLS + DAY_COLS[:5]]))

    # ── ID columns analysis ──
    print(f"\n#### 实体维度统计（仅读取ID列）\n")
    id_df_list = []
    for chunk in pd.read_csv(MAIN_FILE, usecols=ID_COLS, chunksize=10000):
        id_df_list.append(chunk)
    id_df = pd.concat(id_df_list, ignore_index=True)

    print("| 列名 | dtype | n_unique | 样本值 |")
    print("|------|-------|----------|--------|")
    for c in ID_COLS:
        uniq = id_df[c].nunique()
        vals = id_df[c].unique()[:5].tolist()
        print(f"| {c} | {id_df[c].dtype} | {uniq:,} | {vals} |")

    # ── Sales stats on wide table (chunked by rows) ──
    print(f"\n#### 销售值统计（宽表逐chunk读取，chunksize=500行）\n")
    print("**进度：**")
    CHUNKSIZE = 500
    acc = dict(n=0, mn=np.inf, mx=-np.inf, sm=0.0, sq=0.0, zr=0, nr=0, sample=[])
    null_count  = 0
    total_cells = 0
    span_list   = []
    chunk_i = 0

    for chunk in pd.read_csv(MAIN_FILE, usecols=DAY_COLS, chunksize=CHUNKSIZE, dtype='float32'):
        chunk_i += 1
        arr = chunk.values  # shape: (rows, 1941)
        total_cells += arr.size
        mask_notna = ~np.isnan(arr)
        flat = arr[mask_notna].astype(np.float64)
        null_count += arr.size - len(flat)

        if len(flat):
            new_acc = dict(
                n=len(flat),
                mn=float(flat.min()),
                mx=float(flat.max()),
                sm=float(flat.sum()),
                sq=float((flat**2).sum()),
                zr=int((flat == 0).sum()),
                nr=int((flat < 0).sum()),
                sample=flat[np.random.choice(len(flat), min(len(flat), 200), replace=False)].tolist(),
            )
            acc = merge_stats(acc, new_acc)

        # Per-row span (days from first non-nan to last non-nan)
        for row in arr:
            valid_idx = np.where(~np.isnan(row))[0]
            span_list.append(int(valid_idx[-1] - valid_idx[0] + 1) if len(valid_idx) else 0)

        if chunk_i % 10 == 0:
            print(f"  chunk {chunk_i}, rows_done={chunk_i*CHUNKSIZE}")

    print(f"  完成，总cells={total_cells:,}")

    spans = np.array(span_list, dtype=float)
    print(f"\n#### 实体时间跨度统计\n")
    n210, n365, n730 = print_entity_span_stats(spans, total_entities)

    print_sales_stats(acc, null_count, total_cells, "d_1~d_1941")

    # ── Auxiliary tables ──
    print("\n### 第三步：辅助表内容摘要\n")

    # calendar
    cal = pd.read_csv(D6 / "calendar.csv")
    print(f"\n#### calendar.csv  (行数: {len(cal):,})\n")
    print_col_summary(cal)
    print(f"\n**前3行：**")
    print(df_to_md(cal, max_rows=3))
    print(f"\n**时间范围：** {cal['date'].min()} ~ {cal['date'].max()}，共 {len(cal)} 天")

    # sell_prices (chunked)
    print(f"\n#### sell_prices.csv（chunked摘要）\n")
    sp_header = pd.read_csv(D6 / "sell_prices.csv", nrows=0)
    SP_COLS = list(sp_header.columns)
    print(f"**列名：** {SP_COLS}")
    sp_sample = pd.read_csv(D6 / "sell_prices.csv", nrows=5)
    print(f"\n**前5行：**")
    print(df_to_md(sp_sample))

    sp_stats  = {c: {'mn': np.inf, 'mx': -np.inf, 'sm': 0.0, 'n': 0, 'null': 0, 'unique': set()} for c in SP_COLS}
    sp_total  = 0
    for sp_chunk in pd.read_csv(D6 / "sell_prices.csv", chunksize=500_000):
        sp_total += len(sp_chunk)
        for c in SP_COLS:
            s = sp_chunk[c]
            sp_stats[c]['null'] += int(s.isna().sum())
            nn = s.dropna()
            if pd.api.types.is_numeric_dtype(s) and len(nn):
                sp_stats[c]['mn'] = min(sp_stats[c]['mn'], float(nn.min()))
                sp_stats[c]['mx'] = max(sp_stats[c]['mx'], float(nn.max()))
                sp_stats[c]['sm'] += float(nn.sum())
                sp_stats[c]['n']  += len(nn)
            else:
                if len(sp_stats[c]['unique']) < 300:
                    sp_stats[c]['unique'].update(str(v) for v in nn.unique().tolist())

    print(f"\n**总行数：** {sp_total:,}\n")
    print("| 列名 | dtype | null_rate | min | max | mean / n_unique |")
    print("|------|-------|-----------|-----|-----|-----------------|")
    for c in SP_COLS:
        st = sp_stats[c]
        nr = st['null']/sp_total if sp_total else 0
        if st['n'] > 0:
            mean_v = st['sm']/st['n']
            print(f"| {c} | numeric | {nr:.4f} | {st['mn']:.4f} | {st['mx']:.4f} | {mean_v:.4f} |")
        else:
            print(f"| {c} | object | {nr:.4f} | — | — | {len(st['unique'])} unique |")

    # ── Join feasibility ──
    print("\n### 第四步：Join可行性评估\n")
    print("| 辅助表 | Join Key | 新增特征 | 覆盖率 | Leakage风险 |")
    print("|--------|----------|----------|--------|-------------|")
    print("| calendar.csv | d_i ↔ wm_yr_wk / date | weekday, month, year, event_name, snap_XX | 100% | 无（历史信息） |")
    print("| sell_prices.csv | store_id × item_id × wm_yr_wk | sell_price | ~97%（部分item无价格） | 周粒度价格，需按wm_yr_wk对齐，不可用当周未来价格 |")

    # ── RFE ──
    cal_feature_cols = [c for c in cal.columns if c not in ['date', 'd', 'wm_yr_wk']]
    print(f"\n### 第五步：冷启动可行性与RFE特征\n")
    print(f"- **推荐实体单位：** item_id × store_id（即宽表每行）")
    print(f"- **source pool（≥210天）：** {n210:,} 个实体（{n210/total_entities*100:.1f}%）")
    print(f"- **冷启动target建议：** 新引入的item（2015年后仅出现在少数dept的item），或按dept_id分层抽样")
    print(f"- **calendar 特征列：** {cal_feature_cols}")
    print(f"- **sell_price 特征：** sell_price（按 wm_yr_wk 对齐后使用，注意不可前瞻当周平均价）")
    print(f"- **ID特征可用：** dept_id, cat_id, store_id, state_id（作为类别编码特征）")
    print(f"- **常数/高缺失检查：** sell_prices 中不存在的 store×item 对其价格全缺失，需单独过滤或将缺失替换为品类均价")

    time_range = f"{cal['date'].min()} ~ {cal['date'].max()}"
    return dict(
        dataset="D6 M5",
        entity_def="item_id × store_id",
        total_entities=total_entities,
        date_range=time_range,
        unique_dates=len(cal),
        total_rows=total_cells_equiv,
        sales_col="d_1~d_1941",
        zero_rate=f"{acc['zr']/acc['n']:.4f}" if acc['n'] else "N/A",
        neg_rate=f"{acc['nr']/acc['n']:.4f}" if acc['n'] else "N/A",
        null_rate=f"{null_count/total_cells:.4f}",
        n_ge210=n210, pct_ge210=f"{n210/total_entities*100:.1f}%",
        n_ge365=n365, n_ge730=n730,
        aux_tables="calendar,sell_prices",
        granularity="日",
    )


# ──────────────────────────────────────────────
# CROSS-DATASET SUMMARY
# ──────────────────────────────────────────────

def print_summary(results):
    print(f"\n{SEP}")
    print("## 跨数据集对比总览表（D4–D6，与D1–D3扫描报告格式一致）")
    print(SEP)
    fields = [
        ('dataset',        '数据集'),
        ('entity_def',     '实体定义'),
        ('total_entities', '实体总数'),
        ('date_range',     '日期范围'),
        ('unique_dates',   '唯一日期数'),
        ('total_rows',     '等效总行数'),
        ('sales_col',      '销售目标列'),
        ('zero_rate',      '零值率'),
        ('neg_rate',       '负值率'),
        ('null_rate',      '目标列缺失率'),
        ('n_ge210',        '≥210天实体数'),
        ('pct_ge210',      '≥210天占比'),
        ('n_ge365',        '≥365天实体数'),
        ('n_ge730',        '≥730天实体数'),
        ('aux_tables',     '辅助表'),
        ('granularity',    '时间粒度'),
    ]
    header  = "| " + " | ".join(ch for _, ch in fields) + " |"
    sep_row = "|" + "|".join("---" for _ in fields) + "|"
    print(header)
    print(sep_row)
    for r in results:
        row = "| " + " | ".join(str(r.get(f, "N/A")) for f, _ in fields) + " |"
        print(row)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("# 数据集 D4/D5/D6 完整扫描报告")
    print(f"扫描时间：{pd.Timestamp.now()}\n")
    results = []

    print("\n## ── 开始扫描 D4 ──")
    try:
        r4 = scan_d4()
        results.append(r4)
        print("\n✓ D4 扫描完成")
    except Exception as e:
        import traceback
        print(f"[ERROR] D4 扫描失败: {e}")
        traceback.print_exc()

    print("\n## ── 开始扫描 D5 ──")
    try:
        r5 = scan_d5()
        results.append(r5)
        print("\n✓ D5 扫描完成")
    except Exception as e:
        import traceback
        print(f"[ERROR] D5 扫描失败: {e}")
        traceback.print_exc()

    print("\n## ── 开始扫描 D6 ──")
    try:
        r6 = scan_d6()
        results.append(r6)
        print("\n✓ D6 扫描完成")
    except Exception as e:
        import traceback
        print(f"[ERROR] D6 扫描失败: {e}")
        traceback.print_exc()

    print_summary(results)
    print("\n\n**全部扫描完成。**")
