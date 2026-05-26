# coding=utf-8
"""将模型预测结果打包为天池提交 zip。

输入：
  1. 预测分数 CSV：group_folder,anomaly_score
  2. 预测掩码目录：<mask_root>/<category>/<sample_id>/{0..4}_mask.png

输出：
  submission.zip
    ├── submission.csv
    └── predicted_masks/<category>/<sample_id>/{0..4}_mask.png

用法：
    python make_submission.py \
        --scores-csv outputs/scores.csv \
        --mask-root outputs/masks \
        --out-dir outputs/submission \
        --zip my_submission.zip
"""

import argparse
import csv
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


MASK_SIZE = (448, 448)
VIEW_COUNT = 5


def normalize_group_path(group_folder):
    parts = str(group_folder).replace("\\", "/").strip("/").split("/")
    safe_parts = [p for p in parts if p not in ("", ".", "..")]
    if not safe_parts:
        raise ValueError("group_folder 格式非法: %s" % group_folder)
    return Path(*safe_parts)


def copy_mask(src, dst):
    """复制并确保mask为448x448灰度图，缺失则生成全黑mask。"""
    if not os.path.exists(src):
        Image.new("L", MASK_SIZE, 0).save(dst)
        return
    mask = Image.open(src).convert("L")
    if mask.size != MASK_SIZE:
        mask = mask.resize(MASK_SIZE, Image.BILINEAR)
    mask.save(dst)


def build_submission(scores_csv, mask_root, out_dir, zip_path):
    """
    构建提交文件。

    Args:
        scores_csv: 预测分数CSV路径，格式 group_folder,anomaly_score
        mask_root:  预测mask根目录
        out_dir:    输出目录
        zip_path:   输出zip路径（None则不打包）
    """
    scores_csv = Path(scores_csv)
    mask_root = Path(mask_root)
    out_dir = Path(out_dir)

    # 清理旧输出
    if out_dir.exists():
        shutil.rmtree(out_dir)
    pred_mask_dir = out_dir / "predicted_masks"
    pred_mask_dir.mkdir(parents=True, exist_ok=True)

    # 读取分数CSV
    df = pd.read_csv(scores_csv, encoding="utf-8-sig")
    required = {"group_folder", "anomaly_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("%s 缺少必要字段: %s" % (scores_csv, ", ".join(sorted(missing))))
    if df["group_folder"].isnull().any():
        raise ValueError("%s 中 group_folder 存在空值" % scores_csv)
    if df["group_folder"].duplicated().any():
        raise ValueError("%s 中存在重复的 group_folder" % scores_csv)

    # 写 submission.csv
    sub_csv = out_dir / "submission.csv"
    df[["group_folder", "anomaly_score"]].to_csv(
        sub_csv, index=False, encoding="utf-8-sig"
    )

    # 复制mask
    for group_folder in df["group_folder"]:
        group_path = normalize_group_path(group_folder)
        dst_dir = pred_mask_dir / group_path
        dst_dir.mkdir(parents=True, exist_ok=True)
        for view_id in range(VIEW_COUNT):
            mask_name = "%d_mask.png" % view_id
            src = mask_root / group_path / mask_name
            copy_mask(str(src), str(dst_dir / mask_name))

    print("提交目录已生成: %s" % out_dir)
    print("  submission.csv: %d 行" % len(df))
    print("  predicted_masks/: %d 个样本" % len(df))

    # 打包zip
    if zip_path:
        zip_path = Path(zip_path)
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(str(out_dir)):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    arcname = os.path.relpath(fpath, str(out_dir))
                    zf.write(fpath, arcname)
        print("提交zip已打包: %s" % zip_path)


def main():
    parser = argparse.ArgumentParser(description="打包天池提交zip")
    parser.add_argument("--scores-csv", required=True,
                        help="预测分数CSV (group_folder,anomaly_score)")
    parser.add_argument("--mask-root", required=True,
                        help="预测mask根目录")
    parser.add_argument("--out-dir", default="outputs/submission",
                        help="输出目录 (默认: outputs/submission)")
    parser.add_argument("--zip", default="submission.zip",
                        help="输出zip路径 (默认: submission.zip)")
    args = parser.parse_args()

    build_submission(args.scores_csv, args.mask_root, args.out_dir, args.zip)


if __name__ == "__main__":
    main()
