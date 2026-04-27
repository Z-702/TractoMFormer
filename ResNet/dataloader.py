import os
import glob
import pandas as pd
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class PPMIDataset(Dataset):
    def __init__(self, root_dir, csv_path, module="FA1", size=320, val_fold=0, mode="train", transform=None):
        """
        root_dir: 数据根目录 (/data01/zixi/tractoembedding_PPMI_143)
        csv_path: 包含 SUB_ID, DX_GROUP, fold 的 CSV
        module: 模块名 (FA1, density, trace1...)
        size: 图像尺寸 (80, 160, 320)
        val_fold: 哪个 fold 作为验证集
        mode: "train" 或 "val"
        """
        self.root_dir = root_dir
        df = pd.read_csv(csv_path)

        # 根据 fold 选择训练集或验证集
        if mode == "train":
            df = df[df["fold"] != val_fold]
        else:
            df = df[df["fold"] == val_fold]

        # 收集符合条件的文件路径
        pattern = os.path.join(
            root_dir,
            "*",  # SUB_ID
            "tractoembedding",
            "da-full",
            f"*-{module}_CLR_sz{size}.nii.gz"
        )
        all_files = glob.glob(pattern)
        suffix = f"-{module}_CLR_sz{size}.nii.gz"
        file_map = {}
        for f in all_files:
            basename = os.path.basename(f)
            if basename.endswith(suffix):
                sub_id = basename[:-len(suffix)]
                file_map[sub_id] = f       


        # 构建样本 (path, label)
        self.samples = []
        for _, row in df.iterrows():
            sub_id = str(row["SUB_ID"])
            dx = row["DX_GROUP"]
            label = 1 if dx == 1 else 0  # 1=PD, 2=Control(0)
            if sub_id in file_map:
                self.samples.append((file_map[sub_id], label))
            else:
                print(f"⚠️ 文件缺失: SUB_ID={sub_id}, 期望 {module}_CLR_sz{size}.nii.gz")

        self.transform = transform or transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),  # (H,W,3) → (3,H,W)
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = nib.load(path).get_fdata()  # (H,W,3)

        # 归一化到 0~255
        arr = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
        arr = arr.astype(np.uint8)

        pil_img = Image.fromarray(arr)  # (H,W,3) → PIL
        img_tensor = self.transform(pil_img)  # (3,H,W)

        return img_tensor, label

def get_loaders(root_dir, csv_path, module="FA1", size=320, val_fold=0, batch_size=8):
    train_dataset = PPMIDataset(root_dir, csv_path, module=module, size=size, val_fold=val_fold, mode="train")
    val_dataset   = PPMIDataset(root_dir, csv_path, module=module, size=size, val_fold=val_fold, mode="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":
    # 🔹 可更改变量
    root_dir = "/data01/zixi/CNP_tractoembedding"
    csv_path = "/data01/zixi/TractoFormer/TractoFormer-MVIT-main/CNP_150.csv"
    module = "density"        # 可改成 density / trace1 /FA1
    size = 320            # 可改成 80 / 160 / 320
    val_fold = 0          # 哪个 fold 作为验证集
    batch_size = 8
    
    # 用函数生成 DataLoader
    train_loader, val_loader = get_loaders(
        root_dir, csv_path,
        module=module,
        size=size,
        val_fold=val_fold,
        batch_size=batch_size
    )
    
    # 打印样例 batch
    images, labels = next(iter(train_loader))
    print("Image batch shape:", images.shape)  # [B, 3, size, size]
    print("Label batch:", labels)
