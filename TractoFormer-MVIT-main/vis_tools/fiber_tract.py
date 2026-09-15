import os
import h5py
import numpy as np
from dipy.io.vtk import load_vtk_streamlines, save_vtk_streamlines

# ======================================================
# === 可配置区域：修改这里来控制导出的 tract =============
# ======================================================
# 1) 想输出全部 tract → SELECTED_KEYS = ["*"]
# 2) 想输出 AF / ILF / CST → SELECTED_KEYS = ["AF", "ILF", "CST"]
# 3) 想输出所有包含 "Left" 的 tract → SELECTED_KEYS = ["Left"]
# 4) 想输出所有包含 "Temporal" 的 tract → SELECTED_KEYS = ["Temporal"]

SELECTED_KEYS = ["CC"]     # ← 你只要改这里


# ======================================================
# === Path Config ======================================
# ======================================================
sub_id = "sub-50016"
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
        t.decode() if isinstance(t, bytes) else t for t in f["tract_name"][:]
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
    if tract_name not in tract_map:
        tract_map[tract_name] = []
    tract_map[tract_name].append(idx)

print("\n📜 All detected tracts:")
for name in tract_map:
    print("  -", name)


# ======================================================
# === Step 3: 根据 SELECTED_KEYS 过滤 tract =============
# ======================================================
def match_tract(name, keys):
    """返回某个 tract 是否需要导出"""
    if "*" in keys:
        return True  # 全部 tract
    
    name_low = name.lower()
    for k in keys:
        if k.lower() in name_low:  # 模糊匹配
            return True
    return False


filtered_tracts = {
    name: idxs for name, idxs in tract_map.items()
    if match_tract(name, SELECTED_KEYS)
}

print("\n🎯 Selected tracts to export:")
for name, idxs in filtered_tracts.items():
    print(f"  {name:40s} → fibers: {len(idxs)}")


# ======================================================
# === Step 4: Export selected tracts ===================
# ======================================================
out_dir = os.path.join(base_dir, "tracts_vtp")
os.makedirs(out_dir, exist_ok=True)

def save_single_tract(name, indices):
    if len(indices) == 0:
        print(f"⚠️ {name} has 0 fibers, skip.")
        return

    subset = [streamlines[i] for i in indices]
    save_path = os.path.join(out_dir, f"{sub_id}_{name}.vtp")
    save_vtk_streamlines(subset, save_path)
    print(f"✅ Saved: {name:40s} → {len(subset)} fibers → {save_path}")

for name, indices in filtered_tracts.items():
    save_single_tract(name, indices)

print("\n🎨 Export complete! Ready for 3D Slicer visualization.")
