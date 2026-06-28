"""
patch_d3_target_identity.py
────────────────────────────────────────────────────────────────────
给旧版 runner 产出的 D3 结果 CSV 补写 target identity 列。

背景：
  aggregate_d1_d6_results.py 要求每行含 target_entity_id / target_store_id
  之一；旧 run 生成时 runner 尚未写入这些列，但 D3 的 target 始终是
  store=10，所以可以直接用常量补全，无需重跑。

操作：
  1. 读取原 CSV
  2. 在 "method" 列之前插入三列：
       target_entity_id = "10"
       target_store_id  = "10"
       target_item_id   = ""
  3. 备份原文件为 <原名>.bak
  4. 将修补后的数据写回原路径

用法：
  ./.venv/bin/python scripts/patch_d3_target_identity.py
  # 或指定路径：
  ./.venv/bin/python scripts/patch_d3_target_identity.py \
      --csv outputs/runs/20260625_192740_D1_D6_full/d3/results/dataset3_results.csv
"""

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_CSV = (
    "outputs/runs/20260625_192740_D1_D6_full/d3/results/dataset3_results.csv"
)

IDENTITY_COLS = {
    "target_entity_id": "10",
    "target_store_id": "10",
    "target_item_id": "",
}


def patch(csv_path: Path) -> None:
    # ── 1. 读取 ──────────────────────────────────────────────────────
    import csv as _csv

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    print(f"读取完成：{len(rows)} 行，{len(fieldnames)} 列")

    # ── 2. 检查是否已经有 identity 列（幂等保护）──────────────────────
    already_present = [c for c in IDENTITY_COLS if c in fieldnames]
    if already_present:
        print(f"⚠️  以下列已存在，跳过补丁（无需重复操作）：{already_present}")
        return

    # ── 3. 确定插入位置：method 列之前；若 method 不存在则插在最前面 ──
    insert_at = fieldnames.index("method") if "method" in fieldnames else 0
    new_fieldnames = (
        fieldnames[:insert_at]
        + list(IDENTITY_COLS.keys())
        + fieldnames[insert_at:]
    )
    print(
        f"插入位置：索引 {insert_at}（"
        + ("method 列之前" if "method" in fieldnames else "最前面")
        + "）"
    )

    # ── 4. 给每行追加常量值 ───────────────────────────────────────────
    for row in rows:
        for col, val in IDENTITY_COLS.items():
            row[col] = val

    # ── 5. 备份原文件 ────────────────────────────────────────────────
    backup = csv_path.with_suffix(".csv.bak")
    shutil.copy2(csv_path, backup)
    print(f"备份已写入：{backup}")

    # ── 6. 写回 ──────────────────────────────────────────────────────
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅  补丁完成：{csv_path}")
    print(f"    新列数：{len(new_fieldnames)}（原 {len(fieldnames)} + 3）")

    # ── 7. 快速验证：读回前 2 行确认列存在 ───────────────────────────
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        verify_fields = list(reader.fieldnames or [])
        first_row = next(reader, None)

    missing = [c for c in IDENTITY_COLS if c not in verify_fields]
    if missing:
        print(f"❌  验证失败，以下列写入后仍缺失：{missing}", file=sys.stderr)
        sys.exit(1)

    if first_row:
        for col, expected in IDENTITY_COLS.items():
            actual = first_row.get(col, "<missing>")
            status = "✓" if actual == expected else f"✗ 期望 {expected!r} 实际 {actual!r}"
            print(f"    {col} = {actual!r}  {status}")

    print("验证通过 — 可重新运行 aggregate_d1_d6_results.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch D3 results CSV with target identity columns")
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV,
        help="Path to dataset3_results.csv (relative to cwd or absolute)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path

    if not csv_path.exists():
        print(f"❌  文件不存在：{csv_path}", file=sys.stderr)
        sys.exit(1)

    patch(csv_path)


if __name__ == "__main__":
    main()
