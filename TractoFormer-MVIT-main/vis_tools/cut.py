import whitematteranalysis as wma
import numpy as np
import vtk
import os

input_path = "/data01/zixi/tractoembedding_PPMI_143/3104/3104_ILF.vtp"
output_path = "/data01/zixi/tractoembedding_PPMI_143/3104/tracts/3104_ILF_left.vtp"

print("📂 Loading:", input_path)
polydata = wma.io.read_polydata(input_path)
n_lines = polydata.GetNumberOfLines()
print(f"✅ Loaded {n_lines} fibers")

points = polydata.GetPoints()
cell_array = polydata.GetLines()

cell_array.InitTraversal()
id_list = vtk.vtkIdList()
fiber_mask = []

for i in range(n_lines):
    cell_array.GetNextCell(id_list)
    coords = np.array([points.GetPoint(id_list.GetId(j)) for j in range(id_list.GetNumberOfIds())])
    mean_x = np.mean(coords[:, 0])
    fiber_mask.append(mean_x < 0)  # 左半球

selected_count = sum(fiber_mask)
print(f"✅ Selected {selected_count} fibers (x<0)")

polydata_left = wma.filter.mask(polydata, fiber_mask)

# 使用 vtk 自带 writer 确保兼容性
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(output_path)
writer.SetInputData(polydata_left)
writer.SetDataModeToBinary()  # 或 SetDataModeToAscii()，Slicer都能读
writer.Write()

print("💾 Re-saved left fibers in fully compatible VTP format:", output_path)
