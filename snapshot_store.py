"""Yahoo 快照索引：本地补抓与 GitHub 历史分开保存，同日优先正式历史。"""
from pathlib import Path
import csv
import math
from datetime import date

BASE = Path(__file__).resolve().parent
LAGS = ("current", "7daysAgo", "30daysAgo", "60daysAgo", "90daysAgo")


def usable_eps(row):
    """价格返回成功不代表盈利模块完整。"""
    try:
        date.fromisoformat(row["0y_end"])
        values = [row["shares"], row["price"]] + [row[f"{p}_{lag}"] for p in ("0y", "+1y") for lag in LAGS]
        return all(math.isfinite(float(v)) for v in values) and float(row["shares"]) > 0
    except (KeyError, TypeError, ValueError):
        return False


def snapshot_counts(path):
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        return len(rows), sum(usable_eps(r) for r in rows)
    except (OSError, csv.Error):
        return 0, 0


def usable_snapshot(path, minimum):
    _, valid = snapshot_counts(path)
    return valid > 0 and valid >= minimum


def snapshot_paths(base=None, include_local=True):
    root = Path(base) if base is not None else BASE
    paths = {}
    counts = {}
    folders = ("yahoo_local", "yahoo") if include_local else ("yahoo",)
    for folder in folders:
        for path in (root / folder).glob("20*.csv"):
            total, valid = snapshot_counts(path)
            if not total or valid < 0.9 * total:
                continue
            # 正式文件无效或覆盖显著少于同日本地文件时，保留有效的本地补抓。
            if path.name in counts and valid < 0.9 * counts[path.name]:
                continue
            paths[path.name] = str(path)
            counts[path.name] = total
    return [paths[name] for name in sorted(paths)]
