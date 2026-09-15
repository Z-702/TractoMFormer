import os
import h5py
import numpy as np
from dipy.io.vtk import load_vtk_streamlines, save_vtk_streamlines

# ======================================================
# === 可配置区域：选择 tract + 设置 Z 裁剪区域 ============
# ======================================================

SELECTED_KEYS = ["CC, "]     # 选择你要导出的 tract

Z_MIN = -40    # ❗修改：Z 下界（mm）
Z_MAX = 90     # ❗修改：Z 上界（mm）

print(f"📌 Z 裁剪范围: {Z_MIN} ~ {Z_MAX} mm")

# ======================================================
# === Path Config ======================================
# ======================================================
sub_id = "3104"
base_dir = f"/data01/zixi/tractoembedding_PPMI_143/{sub_id}"

vtk_path = f"{base_dir}/tracts/{sub_id}.vtp"
h5_path = f"{base_dir}/tracts/{sub_id}.h5"

print(f"📂 Loading fiber-to-tract mapping from: {h5_path}")

# ======================================================
# === Step 1: Load streamlines =========================
# ======================================================
streamlines = load_vtk_streamlines(vtk_path)

with h5py.File(h5_path, "r") as f:
    tract_ids = np.array(f["tract_list"]).astype(int)
    tract_names = [
        t.decode() if isinstance(t, bytes) else t
        for t in f["tract_name"][:]
    ]

print(f"✅ Streamlines loaded: {len(streamlines)}")
print(f"✅ tract_list count : {len(tract_ids)}")
print(f"✅ tract_name count : {len(tract_names)}")

# ======================================================
# === Step 2: Build tract → fiber index mapping ========
# ======================================================
tract_map = {}

for idx, tract_id in enumerate(tract_ids):
    tract_name = tract_names[tract_id]
    tract_map.setdefault(tract_name, []).append(idx)

print("\n📜 All detected tracts:")
for name in tract_map:
    print("  -", name)

# ======================================================
# === Step 3: 根据 SELECTED_KEYS 过滤 tract =============
# ======================================================
def match_tract(name, keys):
    if "*" in keys:
        return True
    name_low = name.lower()
    for k in keys:
        if k.lower() in name_low:
            return True
    return False

filtered_tracts = {
    name: idxs
    for name, idxs in tract_map.items()
    if match_tract(name, SELECTED_KEYS)
}

print("\n🎯 Selected tracts to export:")
for name, idxs in filtered_tracts.items():
    print(f"  {name:40s} → fibers: {len(idxs)}")

# ======================================================
# === Step 4.5: Z 方向裁剪函数 ===========================
# ======================================================
def crop_by_Z(fiber, zmin, zmax):
    """返回裁剪后的 fiber；若无有效点则返回 None"""
    cropped = [p for p in fiber if zmin <= p[2] <= zmax]
    return cropped if len(cropped) > 1 else None

# ======================================================
# === Step 5: Export selected tracts ===================
# ======================================================
out_dir = os.path.join(base_dir, "tracts_vtp_Zcrop")
os.makedirs(out_dir, exist_ok=True)

def save_single_tract(name, indices):
    print(f"\n✂️ Processing tract: {name}")
    out_streamlines = []

    for i in indices:
        fiber = streamlines[i]
        cropped = crop_by_Z(fiber, Z_MIN, Z_MAX)
        if cropped is not None:
            out_streamlines.append(cropped)

    if len(out_streamlines) == 0:
        print(f"⚠️ All fibers removed by Z cropping! skip {name}")
        return

    save_path = os.path.join(out_dir, f"{sub_id}_{name}_Zcrop.vtp")
    save_vtk_streamlines(out_streamlines, save_path)

    print(f"✅ Saved: {name:40s} → {len(out_streamlines)} fibers → {save_path}")

for name, indices in filtered_tracts.items():
    save_single_tract(name, indices)

print("\n🎨 Export complete with Z cropping!")
