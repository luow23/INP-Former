# coding=utf-8
"""本地指标计算脚本，与天池 evaluate.py 评分口径完全一致。

用法：
    python metrics_local.py --standard-dir standard --submission-dir submit --out metrics.json
    python metrics_local.py --standard-zip standard_A.zip --submission-zip my_submission.zip --out metrics.json
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


MASK_SIZE = (448, 448)
VIEW_COUNT = 5

SEEN_CATEGORIES = {
    "3_adapter", "DVD_switch", "D_sub_connector", "PLCC_socket", "VR_joystick",
    "accurate_detection_switch", "battery", "blade_switch", "boost_converter_module",
    "button_battery_holder", "circuit_breaker", "connector_housing_female",
    "crimp_st_cable_mount_box", "dc_jack", "dc_power_connector", "detection_switch",
    "effect_transistor", "electronic_watch_movement", "ffc_connector_plug",
    "ingot_buckle", "laser_diode", "lego_pin_connector_plate", "limit_switch",
    "lithium_battery_plug", "littel_fuse", "lock", "miniature_lifting_motor",
    "mobile_charging_connector", "motor_bracket", "motor_gear_reducer", "motor_plug",
    "pencil_sharpener", "pinboard_connector", "potentiometer", "power_jack",
    "power_strip_socket", "purple_clay_pot", "retaining_ring", "rheostat",
    "self_lock_switch", "silicon_cell_sensor", "single_switch", "smd_receiver_module",
    "suction_cup", "toy_tire", "travel_switch", "vacuum_switch",
    "vehicle_harness_conductor", "vibration_motor", "wireless_receiver_module",
}


# ========== 工具函数 ==========
def safe_extract(zip_path, dst_dir):
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        dst_abs = os.path.abspath(dst_dir)
        for member in zip_ref.infolist():
            target = os.path.abspath(os.path.join(dst_dir, member.filename))
            if not target.startswith(dst_abs + os.sep) and target != dst_abs:
                raise ValueError("压缩包中包含非法路径: %s" % member.filename)
        zip_ref.extractall(dst_dir)


def find_target_file(base_dir, target_name):
    for root, dirs, files in os.walk(base_dir):
        if target_name in files:
            return os.path.join(root, target_name)
        if target_name in dirs:
            return os.path.join(root, target_name)
    return None


def require_columns(df, required_cols, file_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError("%s 缺少必要字段: %s" % (file_name, ", ".join(missing)))


def read_mask(path, is_prediction):
    if not os.path.exists(path):
        return np.zeros(MASK_SIZE, dtype=np.float32)
    try:
        mask = Image.open(path).convert("L")
    except Exception:
        raise ValueError("无法读取掩码图片: %s" % path)
    if mask.size != MASK_SIZE:
        if is_prediction:
            raise ValueError("预测掩码尺寸必须为 448x448: %s" % path)
        mask = mask.resize(MASK_SIZE, Image.NEAREST)
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    if not np.isfinite(arr).all():
        raise ValueError("预测掩码中存在非法数值: %s" % path)
    return arr


def normalize_group_path(group_folder):
    parts = str(group_folder).replace("\\", "/").strip("/").split("/")
    safe_parts = [p for p in parts if p not in ("", ".", "..")]
    return os.path.join(*safe_parts) if safe_parts else ""


def calc_f1_max(y_true, y_pred):
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    return float(np.max(f1_scores))


def average_metric(cat_metrics, categories, metric_name):
    values = [
        cat_metrics[cat][metric_name]
        for cat in categories
        if metric_name in cat_metrics.get(cat, {})
    ]
    return float(np.mean(values)) if values else 0.0


def average_metrics(cat_metrics, categories, metric_names):
    values = [average_metric(cat_metrics, categories, name) for name in metric_names]
    return float(np.mean(values)) if values else 0.0


def validate_submission(sub_df):
    require_columns(sub_df, ["group_folder", "anomaly_score"], "submission.csv")
    if sub_df["group_folder"].isnull().any():
        raise ValueError("submission.csv 中 group_folder 存在空值")
    if sub_df["group_folder"].duplicated().any():
        raise ValueError("submission.csv 中存在重复的 group_folder")
    sub_df["anomaly_score"] = pd.to_numeric(sub_df["anomaly_score"], errors="coerce")
    if sub_df["anomaly_score"].isnull().any():
        raise ValueError("submission.csv 中 anomaly_score 存在空值或非数字")
    if not np.isfinite(sub_df["anomaly_score"].values).all():
        raise ValueError("submission.csv 中 anomaly_score 存在 NaN 或 Inf")
    return sub_df


# ========== 核心计算 ==========
def calculate_metrics(standard_dir, submission_dir):
    """计算所有指标，与天池 evaluate.py 口径一致。"""
    std = Path(standard_dir)
    sub = Path(submission_dir)

    gt_csv_path = std / "ground_truth.csv"
    gt_mask_dir = std / "masks"
    sub_csv_path = sub / "submission.csv"
    sub_mask_dir = sub / "predicted_masks"

    # 也支持递归查找
    if not gt_csv_path.exists():
        p = find_target_file(standard_dir, "ground_truth.csv")
        if p:
            gt_csv_path = Path(p)
            gt_mask_dir = Path(find_target_file(standard_dir, "masks"))
    if not sub_csv_path.exists():
        p = find_target_file(submission_dir, "submission.csv")
        if p:
            sub_csv_path = Path(p)
            sub_mask_dir = Path(find_target_file(submission_dir, "predicted_masks"))

    if not gt_csv_path.exists():
        raise FileNotFoundError("找不到 ground_truth.csv")
    if gt_mask_dir is None or not Path(gt_mask_dir).exists():
        raise FileNotFoundError("找不到 masks 目录")
    if not sub_csv_path.exists():
        raise FileNotFoundError("找不到 submission.csv")
    if sub_mask_dir is None or not Path(sub_mask_dir).exists():
        raise FileNotFoundError("找不到 predicted_masks 目录")

    gt_df = pd.read_csv(gt_csv_path, encoding="utf-8-sig")
    sub_df = pd.read_csv(sub_csv_path, encoding="utf-8-sig")
    require_columns(gt_df, ["group_folder", "label"], "ground_truth.csv")
    sub_df = validate_submission(sub_df)

    gt_df["label"] = pd.to_numeric(gt_df["label"], errors="coerce")
    if gt_df["label"].isnull().any():
        raise ValueError("ground_truth.csv 中 label 存在空值或非数字")
    if not set(gt_df["label"].unique()).issubset({0, 1}):
        raise ValueError("ground_truth.csv 中 label 必须为 0 或 1")

    merged = pd.merge(gt_df, sub_df, on="group_folder", how="left")
    if merged["anomaly_score"].isnull().any():
        raise ValueError("submission.csv 缺失了部分 group_folder 的预测得分")

    merged["category"] = merged["group_folder"].apply(
        lambda x: str(x).replace("\\", "/").split("/")[0]
    )
    categories = merged["category"].unique()
    cat_metrics = {}

    for cat in categories:
        cat_data = merged[merged["category"] == cat]
        y_true_cls = cat_data["label"].values.astype(np.int8)
        y_pred_cls = cat_data["anomaly_score"].values.astype(np.float32)
        metrics = {}

        # Image-level 指标
        if len(np.unique(y_true_cls)) > 1:
            metrics["I-AUROC"] = float(roc_auc_score(y_true_cls, y_pred_cls))
            metrics["I-AP"] = float(average_precision_score(y_true_cls, y_pred_cls))

        # Pixel-level 指标
        cat_gt_masks, cat_pred_masks = [], []
        for group in cat_data["group_folder"]:
            safe_group = normalize_group_path(group)
            if not safe_group:
                raise ValueError("group_folder 格式非法: %s" % group)
            for i in range(VIEW_COUNT):
                mask_name = "%d_mask.png" % i
                gt_path = os.path.join(str(gt_mask_dir), safe_group, mask_name)
                pred_path = os.path.join(str(sub_mask_dir), safe_group, mask_name)
                cat_gt_masks.append(read_mask(gt_path, is_prediction=False).reshape(-1))
                cat_pred_masks.append(read_mask(pred_path, is_prediction=True).reshape(-1))

        y_true_px = np.concatenate(cat_gt_masks).astype(np.int8)
        y_pred_px = np.concatenate(cat_pred_masks).astype(np.float32)

        if len(np.unique(y_true_px)) > 1:
            metrics["P-AUROC"] = float(roc_auc_score(y_true_px, y_pred_px))
            metrics["P-AP"] = float(average_precision_score(y_true_px, y_pred_px))
            metrics["P-F1max"] = float(calc_f1_max(y_true_px, y_pred_px))
        cat_metrics[cat] = metrics

    # 按类别聚合
    seen_cats = [c for c in categories if c in SEEN_CATEGORIES]
    zs_cats = [c for c in categories if c not in SEEN_CATEGORIES]
    cls_cats = seen_cats if zs_cats else categories

    s_i_roc = average_metric(cat_metrics, cls_cats, "I-AUROC")
    s_i_ap = average_metric(cat_metrics, cls_cats, "I-AP")
    s_p_roc = average_metric(cat_metrics, categories, "P-AUROC")
    s_p_ap = average_metric(cat_metrics, categories, "P-AP")
    s_p_f1 = average_metric(cat_metrics, categories, "P-F1max")

    s_cls = average_metrics(cat_metrics, cls_cats, ["I-AUROC", "I-AP"])
    s_seg = average_metrics(cat_metrics, categories, ["P-AUROC", "P-AP", "P-F1max"])

    if zs_cats:
        zs_i_roc = average_metric(cat_metrics, zs_cats, "I-AUROC")
        zs_i_ap = average_metric(cat_metrics, zs_cats, "I-AP")
        zs_p_roc = average_metric(cat_metrics, zs_cats, "P-AUROC")
        zs_p_ap = average_metric(cat_metrics, zs_cats, "P-AP")
        zs_p_f1 = average_metric(cat_metrics, zs_cats, "P-F1max")
        s_zs = float(np.mean([zs_i_roc, zs_i_ap, zs_p_roc, zs_p_ap, zs_p_f1]))
        final_score = float(max(0.0, (0.3 * s_cls + 0.5 * s_seg + 0.2 * s_zs) * 100.0))
    else:
        zs_i_roc = zs_i_ap = zs_p_roc = zs_p_ap = zs_p_f1 = s_zs = 0.0
        final_score = float(max(0.0, (0.3 * s_cls + 0.7 * s_seg) * 100.0))

    result = {
        "score": final_score,
        "I-AUROC": s_i_roc,
        "I-AP": s_i_ap,
        "P-AUROC": s_p_roc,
        "P-AP": s_p_ap,
        "P-F1max": s_p_f1,
        "S_cls": s_cls,
        "S_seg": s_seg,
        "S_zs": s_zs,
        "seen_category_count": len(seen_cats),
        "zero_shot_category_count": len(zs_cats),
    }

    if zs_cats:
        result.update({
            "ZS-I-AUROC": zs_i_roc,
            "ZS-I-AP": zs_i_ap,
            "ZS-P-AUROC": zs_p_roc,
            "ZS-P-AP": zs_p_ap,
            "ZS-P-F1max": zs_p_f1,
        })

    return result


# ========== 命令行入口 ==========
def main():
    parser = argparse.ArgumentParser(description="本地指标计算（与天池 evaluate.py 口径一致）")
    parser.add_argument("--standard-dir", default="",
                        help="标准答案目录（含 ground_truth.csv 和 masks/）")
    parser.add_argument("--submission-dir", default="",
                        help="提交目录（含 submission.csv 和 predicted_masks/）")
    parser.add_argument("--standard-zip", default="",
                        help="标准答案 zip（自动解压）")
    parser.add_argument("--submission-zip", default="",
                        help="提交 zip（自动解压）")
    parser.add_argument("--out", default="metrics.json", help="输出结果文件路径")
    args = parser.parse_args()

    tmp_dirs = []

    # 处理标准答案
    if args.standard_zip:
        tmp = tempfile.mkdtemp(prefix="std_")
        tmp_dirs.append(tmp)
        safe_extract(args.standard_zip, tmp)
        std_dir = tmp
    elif args.standard_dir:
        std_dir = args.standard_dir
    else:
        print("错误: 需要指定 --standard-dir 或 --standard-zip")
        sys.exit(1)

    # 处理提交文件
    if args.submission_zip:
        tmp = tempfile.mkdtemp(prefix="sub_")
        tmp_dirs.append(tmp)
        safe_extract(args.submission_zip, tmp)
        sub_dir = tmp
    elif args.submission_dir:
        sub_dir = args.submission_dir
    else:
        print("错误: 需要指定 --submission-dir 或 --submission-zip")
        sys.exit(1)

    try:
        result = calculate_metrics(std_dir, sub_dir)

        text = json.dumps(result, ensure_ascii=False, indent=2)
        print(text)

        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
            print("\n结果已保存到: %s" % args.out)

    finally:
        for d in tmp_dirs:
            if os.path.isdir(d):
                shutil.rmtree(d)


if __name__ == "__main__":
    main()
