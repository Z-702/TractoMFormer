import os
import h5py
import numpy as np
import nibabel as nib
from dipy.io.vtk import load_vtk_streamlines

# ======== 配置区域 ========
sub_id = "3104"
base_dir = f"/data01/zixi/tractoembedding_PPMI_143/{sub_id}"

embed_ref_path = f"{base_dir}/tractoembedding/da-full/3104-FA1_CLR_sz320.nii.gz"  # reference grid
vtk_path = f"{base_dir}/tracts/{sub_id}.vtp"
h5_path = f"{base_dir}/tracts/{sub_id}.h5"

TRACT_KEYS = {
    "AF": ["AF"],
    "CST": ["CST"],
    "ILF": ["ILF"],
}

# ======== Step 1: Load reference grid ========
ref_img = nib.load(embed_ref_path)
ref_data = ref_img.get_fdata()
aff = ref_img.affine
shape = ref_data.shape
print("Reference grid:", shape)

# 空 mask
mask_AF  = np.zeros(shape, dtype=np.int16)
mask_CST = np.zeros(shape, dtype=np.int16)
mask_ILF = np.zeros(shape, dtype=np.int16)

# ======== Step 2: Load fibers ========
print("Loading streamlines...")
streamlines = load_vtk_streamlines(vtk_path)

with h5py.File(h5_path, "r") as f:
    tract_ids = np.array(f["tract_list"]).astype(int)
    tract_names = [
        t.decode() if isinstance(t, bytes) else t
        for t in f["tract_name"][:]
    ]

# tract_name → fiber index list
tract_map = {}
for idx, t_id in enumerate(tract_ids):
    tname = tract_names[t_id]
    tract_map.setdefault(tname, []).append(idx)

print("Available tracts:", list(tract_map.keys()))

# ======== Step 3: Voxelization helper ========
def world_to_voxel(coords, affine):
    M = np.linalg.inv(affine)
    voxel = (M @ np.c_[coords, np.ones(coords.shape[0])].T).T[:, :3]
    return np.round(voxel).astype(int)

# ======== Step 4: Process AF / CST / ILF ========
def fill_mask(mask, fibers):
    for sl in fibers:
        vox = world_to_voxel(sl, aff)
        for x, y, z in vox:
            if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
                mask[x, y, z] = 1

def match(name, keys):
    name_low = name.lower()
    for k in keys:
        if k.lower() in name_low:
            return True
    return False

# 获取 AF/CST/ILF fiber 索引
fib_AF = [idx for name, ids in tract_map.items() if match(name, TRACT_KEYS["AF"]) for idx in ids]
fib_CST = [idx for name, ids in tract_map.items() if match(name, TRACT_KEYS["CST"]) for idx in ids]
fib_ILF = [idx for name, ids in tract_map.items() if match(name, TRACT_KEYS["ILF"]) for idx in ids]

print("AF fibers:", len(fib_AF))
print("CST fibers:", len(fib_CST))
print("ILF fibers:", len(fib_ILF))

# 填充掩膜
fill_mask(mask_AF,  [streamlines[i] for i in fib_AF])
fill_mask(mask_CST, [streamlines[i] for i in fib_CST])
fill_mask(mask_ILF, [streamlines[i] for i in fib_ILF])

# ======== Step 5: Save output masks ========
outdir = f"{base_dir}/tractoembedding/da-full"

nib.save(nib.Nifti1Image(mask_AF,  aff, ref_img.header), f"{outdir}/3104_embed_AF.nii.gz")
nib.save(nib.Nifti1Image(mask_CST, aff, ref_img.header), f"{outdir}/3104_embed_CST.nii.gz")
nib.save(nib.Nifti1Image(mask_ILF, aff, ref_img.header), f"{outdir}/3104_embed_ILF.nii.gz")

print("\n🎉 Done! Generated embedding masks:")
print(f" - {outdir}/3104_embed_AF.nii.gz")
print(f" - {outdir}/3104_embed_CST.nii.gz")
print(f" - {outdir}/3104_embed_ILF.nii.gz")
