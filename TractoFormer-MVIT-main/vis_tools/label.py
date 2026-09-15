import nibabel as nib
import numpy as np
import os

# === 输入路径 ===
base_dir = "/data01/zixi/tractoembedding_PPMI_143/3104/tractoembedding/da-full"
files = {
    1: os.path.join(base_dir, "3104_embed_AF.nii.gz"),
    2: os.path.join(base_dir, "3104_embed_CST.nii.gz"),
    3: os.path.join(base_dir, "3104_embed_ILF.nii.gz"),
}

# === 输出路径 ===
output_labelmap = os.path.join(base_dir, "3104_embed_label.nii.gz")

# === 读取第一个 embed 图像作为参考 ===
ref_img = nib.load(list(files.values())[0])
ref_data = ref_img.get_fdata()
label_array = np.zeros(ref_data.shape, dtype=np.int16)

# === 逐个 tract 保存单独 mask + 构建 labelmap ===
for label_value, file_path in files.items():
    print(f"Processing {file_path} → label {label_value}")

    # 读取 embed
    img = nib.load(file_path)
    data = img.get_fdata()

    # 生成二值 mask
    mask = (data > 0).astype(np.int16)

    # === 保存单独 mask ===
    output_single_mask = os.path.join(base_dir, f"3104_embed_label_{label_value}.nii.gz")
    nib.save(
        nib.Nifti1Image(mask, affine=ref_img.affine, header=ref_img.header),
        output_single_mask
    )
    print(f"  → Saved mask: {output_single_mask}")

    # === 写入 labelmap（AF=1, CST=2, ILF=3）===
    label_array[mask > 0] = label_value

# === 保存合并 labelmap ===
merged_img = nib.Nifti1Image(label_array, ref_img.affine, ref_img.header)
nib.save(merged_img, output_labelmap)

print(f"\n🎉 All Done!")
print(f"👉 Saved merged labelmap: {output_labelmap}")
