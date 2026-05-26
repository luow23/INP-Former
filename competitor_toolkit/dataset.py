# coding=utf-8
"""数据集加载模块

支持从以下结构加载数据：
  标准答案(ground truth):
    standard/
      ground_truth.csv        # group_folder,label
      masks/<cat>/<sid>/{0..4}_mask.png

  预测结果:
    submit/
      submission.csv          # group_folder,anomaly_score
      predicted_masks/<cat>/<sid>/{0..4}_mask.png

也支持直接从 zip 文件加载（自动解压到临时目录）。
"""

import os
import csv
import zipfile
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


# ========== 常量 ==========
MASK_SIZE = (448, 448)
VIEW_COUNT = 5  # 每个样本5个视角

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


def safe_extract(zip_path, dst_dir):
    """安全解压zip，防止路径穿越攻击。"""
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
    """递归查找文件或目录（允许多打包一层文件夹）。"""
    for root, dirs, files in os.walk(base_dir):
        if target_name in files:
            return os.path.join(root, target_name)
        if target_name in dirs:
            return os.path.join(root, target_name)
    return None


# ========== 数据集加载 ==========
class AnomalyDataset:
    """异常检测数据集加载器。

    用法：
        # 从目录加载
        ds = AnomalyDataset.from_dir("./standard")

        # 从zip加载（自动解压）
        ds = AnomalyDataset.from_zip("standard_A.zip")

        # 加载某个样本的图片
        images = ds.load_images("3_adapter/S0001")
        # images.shape == (5, 448, 448)  5个视角的灰度图

        # 获取标签
        label = ds.get_label("3_adapter/S0001")  # 0 或 1

        # 获取某个类别的所有样本
        samples = ds.get_category_samples("3_adapter")
    """

    def __init__(self, root_dir):
        """
        Args:
            root_dir: 包含 ground_truth.csv 和 masks/ 的目录路径
        """
        self.root_dir = Path(root_dir)
        self.gt_csv = self.root_dir / "ground_truth.csv"
        self.mask_dir = self.root_dir / "masks"

        if not self.gt_csv.exists():
            raise FileNotFoundError("找不到 ground_truth.csv: %s" % self.gt_csv)
        if not self.mask_dir.exists():
            raise FileNotFoundError("找不到 masks 目录: %s" % self.mask_dir)

        # 加载标签
        self._labels = {}          # group_folder -> label (0/1)
        self._categories = {}      # group_folder -> category
        self._category_samples = {}  # category -> [group_folder, ...]

        with open(self.gt_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gf = row["group_folder"].strip()
                label = int(row["label"])
                cat = gf.split("/")[0]
                self._labels[gf] = label
                self._categories[gf] = cat
                self._category_samples.setdefault(cat, []).append(gf)

    @classmethod
    def from_dir(cls, root_dir):
        """从解压后的目录加载。"""
        return cls(root_dir)

    @classmethod
    def from_zip(cls, zip_path, extract_dir=None):
        """从zip文件加载（自动解压）。"""
        if extract_dir is None:
            extract_dir = tempfile.mkdtemp(prefix="anomaly_ds_")
        safe_extract(zip_path, extract_dir)
        actual = find_target_file(extract_dir, "ground_truth.csv")
        if actual:
            return cls(os.path.dirname(actual))
        return cls(extract_dir)

    # ---------- 基本信息 ----------
    @property
    def categories(self):
        """所有类别列表。"""
        return sorted(self._category_samples.keys())

    @property
    def all_samples(self):
        """所有样本的 group_folder 列表。"""
        return list(self._labels.keys())

    def get_label(self, group_folder):
        """获取某个样本的标签 (0=正常, 1=异常)。"""
        return self._labels[group_folder]

    def get_category_samples(self, category):
        """获取某个类别下所有样本的 group_folder 列表。"""
        return self._category_samples.get(category, [])

    # ---------- 图片加载 ----------
    def load_images(self, group_folder):
        """加载某个样本的5个视角图片。

        Returns:
            np.ndarray, shape=(5, 448, 448), dtype=float32, 值域[0,1]
        """
        group_path = group_folder.replace("\\", "/").strip("/")
        imgs = []
        for i in range(VIEW_COUNT):
            mask_file = self.mask_dir / group_path / ("%d_mask.png" % i)
            imgs.append(self._read_image(mask_file))
        return np.stack(imgs, axis=0)

    def load_mask(self, group_folder, view_id):
        """加载单张mask图片。

        Args:
            group_folder: 如 "3_adapter/S0001"
            view_id: 0~4

        Returns:
            np.ndarray, shape=(448, 448), dtype=float32, 值域[0,1]
        """
        group_path = group_folder.replace("\\", "/").strip("/")
        mask_file = self.mask_dir / group_path / ("%d_mask.png" % view_id)
        return self._read_image(mask_file)

    def _read_image(self, path):
        if not path.exists():
            return np.zeros(MASK_SIZE, dtype=np.float32)
        img = Image.open(path).convert("L")
        if img.size != MASK_SIZE:
            img = img.resize(MASK_SIZE, Image.NEAREST)
        return np.asarray(img, dtype=np.float32) / 255.0


class SubmissionBuilder:
    """构建提交结果。

    用法：
        builder = SubmissionBuilder()

        # 添加预测结果
        builder.add_sample("3_adapter/S0001", anomaly_score=0.12,
                           masks=[mask0, mask1, mask2, mask3, mask4])

        # 生成提交zip
        builder.save("outputs/submission", zip_path="my_submission.zip")
    """

    def __init__(self):
        self.samples = []  # [(group_folder, anomaly_score, [5 masks])]

    def add_sample(self, group_folder, anomaly_score, masks=None):
        """添加一个样本的预测结果。

        Args:
            group_folder: 如 "3_adapter/S0001"
            anomaly_score: 异常分数，越大越异常
            masks: 5个视角的mask，每个是 (448,448) 的numpy数组或None
                   None表示全黑mask
        """
        if masks is None:
            masks = [np.zeros(MASK_SIZE, dtype=np.float32)] * VIEW_COUNT
        assert len(masks) == VIEW_COUNT, "每个样本需要5个视角的mask"
        self.samples.append((group_folder, float(anomaly_score), masks))

    def save(self, out_dir, zip_path=None):
        """保存为提交目录结构，可选打包为zip。"""
        out = Path(out_dir)
        if out.exists():
            shutil.rmtree(out)
        pred_dir = out / "predicted_masks"
        pred_dir.mkdir(parents=True, exist_ok=True)

        # 写 submission.csv
        csv_path = out / "submission.csv"
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["group_folder", "anomaly_score"])
            for gf, score, _ in self.samples:
                writer.writerow([gf, score])

        # 写 mask 图片
        for gf, _, masks in self.samples:
            group_path = gf.replace("\\", "/").strip("/")
            mask_dir = pred_dir / group_path
            mask_dir.mkdir(parents=True, exist_ok=True)
            for i, mask in enumerate(masks):
                if mask.dtype != np.float32:
                    mask = mask.astype(np.float32)
                # 归一化到[0,255]
                mask_uint8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
                img = Image.fromarray(mask_uint8, mode="L")
                if img.size != MASK_SIZE:
                    img = img.resize(MASK_SIZE, Image.BILINEAR)
                img.save(mask_dir / ("%d_mask.png" % i))

        print("提交文件已保存到: %s" % out)

        # 打包zip
        if zip_path:
            if os.path.exists(zip_path):
                os.remove(zip_path)
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(str(out)):
                    for fname in files:
                        fpath = os.path.join(root, fname)
                        arcname = os.path.relpath(fpath, str(out))
                        zf.write(fpath, arcname)
            print("提交zip已打包: %s" % zip_path)


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 1. 从目录加载数据集
    ds = AnomalyDataset.from_dir("./standard")
    print("类别数: %d" % len(ds.categories))
    print("总样本数: %d" % len(ds.all_samples))
    print("类别列表: %s" % ds.categories[:5])

    # 2. 加载一个样本的图片
    sample = ds.all_samples[0]
    images = ds.load_images(sample)
    print("\n样本 %s 的图片shape: %s" % (sample, images.shape))
    print("标签: %d" % ds.get_label(sample))

    # 3. 构建一个假的提交（全零预测）
    builder = SubmissionBuilder()
    for gf in ds.all_samples:
        masks = [np.zeros(MASK_SIZE, dtype=np.float32) for _ in range(VIEW_COUNT)]
        builder.add_sample(gf, anomaly_score=0.0, masks=masks)
    builder.save("outputs/demo_submission", zip_path="demo_submission.zip")
