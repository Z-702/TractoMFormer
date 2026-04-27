import torch
from torch.utils.data import Dataset
from torchvision import transforms
from DTI.utils import read_csv, export
import os
import numpy as np
import logging
import math
import csv

LOG = logging.getLogger('dataset')


def count_list(alist):
    data_dict = {}
    for key in alist:
        data_dict[key] = data_dict.get(key, 0) + 1
    return data_dict


commissural_clusters = [
    4, 9, 34, 41, 47, 53, 57, 58, 63, 69, 70, 74, 87, 92, 110, 111, 115, 143, 145, 146, 147, 160, 164,
    251, 252, 253, 258, 263, 264, 269, 272, 306, 312, 315, 323, 331, 335, 339, 343, 351, 364, 372, 376,
    404, 411, 438, 441, 449, 457, 466, 469, 476, 485, 486, 489, 520, 523, 526, 544, 546, 550, 558, 577,
    578, 583, 588, 592, 598, 602, 615, 621, 624, 628, 634, 646, 654, 659, 664, 671, 678, 684, 703, 771, 782
]


def _normalize_name(name):
    s = str(name).strip()
    s = s.replace(" ", "")
    s = s.replace("_", "").replace("-", "").replace(".", "")
    return s.lower()


def _safe_float(x):
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in ["nan", "none", "null", "na"]:
        return np.nan
    try:
        return float(s)
    except:
        return np.nan


def _find_first_existing_column(fieldnames, candidates):
    normalized_map = {_normalize_name(name): name for name in (fieldnames or [])}
    for candidate in candidates:
        matched = normalized_map.get(_normalize_name(candidate))
        if matched:
            return matched
    return None


def _resolve_label_column(opt, fieldnames):
    explicit_label_col = getattr(opt, "label_col", "") or getattr(opt, "LABEL_COL", "")
    if explicit_label_col:
        if explicit_label_col not in fieldnames:
            raise KeyError(f"指定的标签列 `{explicit_label_col}` 不在 demographics CSV 中。可用列: {fieldnames}")
        return explicit_label_col

    legacy_dx_col = getattr(opt, "dx_col", None) or getattr(opt, "DX_COL", None)
    target = str(getattr(opt, "target", "")).strip().lower()

    if legacy_dx_col:
        if legacy_dx_col not in fieldnames:
            raise KeyError(f"指定的标签列 `{legacy_dx_col}` 不在 demographics CSV 中。可用列: {fieldnames}")
        return legacy_dx_col

    if target == "sex":
        sex_candidates = ["sex", "SEX", "Gender", "gender", "Sex"]
        matched = _find_first_existing_column(fieldnames, sex_candidates)
        if matched is not None:
            return matched
        raise KeyError(
            "target=sex，但 demographics CSV 中没有找到性别列。"
            f"请使用 --label-col 显式指定。可用列: {fieldnames}"
        )

    if target == "age":
        age_candidates = ["age", "AGE", "Age"]
        matched = _find_first_existing_column(fieldnames, age_candidates)
        if matched is None:
            raise KeyError(
                "target=age，但 demographics CSV 中没有找到年龄列。"
                f"请使用 --label-col 显式指定。可用列: {fieldnames}"
            )
        return matched

    raise KeyError(f"无法为 target={getattr(opt, 'target', None)} 推断标签列，请使用 --label-col 显式指定。")


def _parse_target_value(raw_value, target):
    target = str(target).strip().lower()

    if target == "sex":
        value = str(raw_value).strip()
        normalized = _normalize_name(value)
        mapping = {
            "0": 0,
            "f": 0,
            "female": 0,
            "woman": 0,
            "girl": 0,
            "m": 1,
            "male": 1,
            "man": 1,
            "boy": 1,
        }
        if normalized in mapping:
            return mapping[normalized]

        numeric = _safe_float(raw_value)
        if np.isnan(numeric):
            raise ValueError(f"无法解析 sex 标签值: {raw_value}")
        numeric = int(numeric)
        if numeric in (1, 2):
            return numeric - 1
        if numeric not in (0, 1):
            raise ValueError(f"sex 标签必须是二分类(0/1)，当前值为: {raw_value}")
        return numeric

    if target == "age":
        numeric = _safe_float(raw_value)
        if np.isnan(numeric):
            raise ValueError(f"无法解析 age 标签值: {raw_value}")
        return float(numeric)

    numeric = _safe_float(raw_value)
    if np.isnan(numeric):
        raise ValueError(f"无法解析标签值: {raw_value}")
    numeric = int(numeric)
    return numeric if numeric in (0, 1) else numeric - 1


@export
def get_hcp_s1200(opt, mode='train', n_fold=0):
    """
    直接根据 demographics_csv 里的 fold 列划分 train / val / test。
    """
    root_dir = opt.data_path
    demo_csv = getattr(opt, "demographics_csv", "") or getattr(opt, "DEMOGRAPHICS_CSV", "")
    subj_col = getattr(opt, "subject_col", None) or "SUB_ID"

    if not demo_csv or not os.path.exists(demo_csv):
        raise FileNotFoundError("❌ 未提供有效的 --demographics-csv（或 DEMOGRAPHICS_CSV）")

    rows = []
    with open(demo_csv, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        label_col = _resolve_label_column(opt, fieldnames)

        has_fold = "fold" in fieldnames

        for r in reader:
            sid = str(r.get(subj_col, "")).strip()
            if not sid:
                continue

            try:
                label_value = _parse_target_value(r.get(label_col, ""), getattr(opt, "target", ""))
            except Exception as exc:
                raise ValueError(f"subject {sid} has invalid value in label column `{label_col}`: {exc}") from exc

            fold_value = None
            if has_fold:
                try:
                    fold_value = int(float(str(r.get("fold", "")).strip()))
                except:
                    fold_value = None

            # 按 fold 划分
            if has_fold and fold_value is not None:
                if mode == 'train' and fold_value == n_fold:
                    continue
                if mode in ['val', 'test'] and fold_value != n_fold:
                    continue

            rows.append([sid, 0, label_value, 0, 0, 1])  # [sid, age, dx, race, hand, bmi]

    data = rows

    # 为每个 subject 找对应的4个特征文件
    for i in reversed(range(len(data))):
        sid = str(data[i][0])

        a1 = os.path.join(root_dir, sid, 'AnatomicalTracts', 'diffusion_measurements_anatomical_tracts.csv')
        c2 = os.path.join(root_dir, sid, 'FiberClustering', 'SeparatedClusters', 'diffusion_measurements_commissural.csv')
        l3 = os.path.join(root_dir, sid, 'FiberClustering', 'SeparatedClusters', 'diffusion_measurements_left_hemisphere.csv')
        r4 = os.path.join(root_dir, sid, 'FiberClustering', 'SeparatedClusters', 'diffusion_measurements_right_hemisphere.csv')

        if not (os.path.exists(a1) and os.path.exists(c2) and os.path.exists(l3) and os.path.exists(r4)):
            a1 = os.path.join(root_dir, 'UKF_2T_AtlasSpace', 'anatomical_tracts', sid + '.csv')
            c2 = os.path.join(root_dir, 'UKF_2T_AtlasSpace', 'tracts_commissural', sid + '.csv')
            l3 = os.path.join(root_dir, 'UKF_2T_AtlasSpace', 'tracts_left_hemisphere', sid + '.csv')
            r4 = os.path.join(root_dir, 'UKF_2T_AtlasSpace', 'tracts_right_hemisphere', sid + '.csv')

        if os.path.exists(a1) and os.path.exists(c2) and os.path.exists(l3) and os.path.exists(r4):
            data[i].extend([a1, c2, l3, r4])
        else:
            data.pop(i)

    transform = transforms.Normalize(mean=0.05, std=0.5)

    LOG.info(f"[{mode}] n_fold={n_fold}, total usable subjects={len(data)}")
    LOG.info(f"[{mode}] n_fold={n_fold}, label_col={label_col}, label_distribution={count_list([row[2] for row in data])}")

    return {
        'root_dir': root_dir,
        'data_list': data,
        'transform': transform
    }


@export
class CreateDataset(Dataset):
    def __init__(self, opt, dataset):
        """
        dataset 已经在 get_hcp_s1200() 阶段按 fold 切好了。
        """
        self.root_dir = dataset['root_dir']
        self.data_list = dataset['data_list']
        self.transform = dataset['transform']
        self.opt = opt

        self.hemispheres = []
        if 'right-hemisphere' in self.opt.HEMISPHERES:
            self.hemispheres.append(-1)
        if 'left-hemisphere' in self.opt.HEMISPHERES:
            self.hemispheres.append(-2)
        if 'commissural' in self.opt.HEMISPHERES:
            self.hemispheres.append(-3)
        if 'anatomical' in self.opt.HEMISPHERES:
            self.hemispheres.append(-4)

        self.features = self._parse_input_features(self.opt.INPUT_FEATURES)

        if len(self.features) == 0:
            raise ValueError(
                f"❌ 没有识别到任何有效 INPUT_FEATURES: {self.opt.INPUT_FEATURES}\n"
                f"支持示例：FA1-mean / FA2-mean / trace1.Mean / trace2.Mean / Num_Fibers / Num_Points"
            )

    def _parse_input_features(self, input_features):
        idx_map = {
            "numpoints": 0,
            "numfibers": 1,
            "fa1mean": 4,
            "fa2mean": 5,
            "trace1mean": 7,
            "trace2mean": 8,
        }

        features = []
        for feat in input_features:
            key = _normalize_name(feat)
            if key in idx_map:
                features.append(idx_map[key])

        return features

    def __len__(self):
        return len(self.data_list)

    def _load_csv_as_numeric_matrix(self, csv_path):
        raw_data = read_csv(csv_path)

        if len(raw_data) <= 1:
            raise ValueError(f"❌ 空 csv 或格式错误: {csv_path}")

        header = [str(h).strip() for h in raw_data[0]]

        if "cluster_id" in header:
            cluster_idx = header.index("cluster_id")
            header = header[:cluster_idx] + header[cluster_idx + 1:]
            body = [row[:cluster_idx] + row[cluster_idx + 1:] for row in raw_data[1:]]
        else:
            body = raw_data[1:]

        clean_rows = []
        for row in body:
            row = list(row)
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            elif len(row) > len(header):
                row = row[:len(header)]

            numeric_row = []
            for v in row[1:]:
                numeric_row.append(_safe_float(v))
            clean_rows.append(numeric_row)

        data = np.array(clean_rows, dtype=np.float32).transpose()
        return data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        all_data = None

        for hemi in self.hemispheres:
            raw_path = self.data_list[idx][hemi]
            data = self._load_csv_as_numeric_matrix(raw_path)

            if hemi in [-1, -2]:
                valid_delete = [i - 1 for i in commissural_clusters if 0 <= i - 1 < data.shape[1]]
                if len(valid_delete) > 0:
                    data = np.delete(data, np.array(valid_delete), axis=1)

            elif hemi == -3:
                not_commissural = [i + 1 for i in range(800) if i + 1 not in commissural_clusters]
                valid_delete = [i - 1 for i in not_commissural if 0 <= i - 1 < data.shape[1]]
                if len(valid_delete) > 0:
                    data = np.delete(data, np.array(valid_delete), axis=1)

            elif hemi == -4:
                keep_idx = list(range(0, 7)) + list(range(8, 10))
                keep_idx = [i for i in keep_idx if i < data.shape[0]]
                data = data[keep_idx, :]

            if all_data is None:
                all_data = data
            else:
                all_data = np.concatenate((all_data, data), axis=1)

        if all_data is None:
            raise ValueError("❌ all_data 为空，请检查输入 csv 和 HEMISPHERES 设置")

        all_data = np.nan_to_num(all_data, nan=0.0, posinf=0.0, neginf=0.0)

        selected_features = []
        for i in self.features:
            if i >= all_data.shape[0]:
                raise IndexError(
                    f"❌ 特征索引 {i} 超出 all_data 行数 {all_data.shape[0]}，请检查 csv 列结构"
                )
            feature = all_data[i, :][None]
            fmin = feature.min()
            fmax = feature.max()
            if abs(fmax - fmin) < 1e-8:
                feature = np.zeros_like(feature, dtype=np.float32)
            else:
                feature = (feature - fmin) / (fmax - fmin + 1e-8)
            selected_features.append(feature)

        x = np.concatenate(selected_features, axis=0).astype(np.float32)

        temp = np.zeros([len(self.features), 3, 800], dtype=np.float32)
        if self.opt.MODEL == '1.5D':
            for i in range(len(self.features)):
                if x.shape[1] >= 716:
                    temp[i, 0, 0:716] = x[i, 0:716]
                if x.shape[1] >= 1432:
                    temp[i, 1, 0:716] = x[i, 716:1432]
                if x.shape[1] > 1432:
                    right_len = min(800 - 716, x.shape[1] - 1432)
                    temp[i, 2, 716:716 + right_len] = x[i, 1432:1432 + right_len]
            x = temp

        if self.opt.MODEL in ['2D-CNN', 'Lenet']:
            dim0, dim1 = x.shape
            size = math.ceil(dim1 ** 0.5)
            x = np.concatenate((x, np.zeros((dim0, size ** 2 - dim1), dtype=np.float32)), axis=1)
            x = x.reshape((dim0, size, size))

        x = torch.from_numpy(x).float()
        y = torch.tensor(self.data_list[idx][2]).float()

        return {'x': x, 'y': y}
