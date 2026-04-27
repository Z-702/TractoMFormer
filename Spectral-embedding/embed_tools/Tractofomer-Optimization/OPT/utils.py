import whitematteranalysis as wma
import numpy as np
import fibers
import os


def convert_fiber_to_array(inputFile, numberOfFibers, fiberLength, numberOfFiberPoints, preproces=False, data='HCP'):
    if not os.path.exists(inputFile):
        print("<wm_cluster_from_atlas.py> Error: Input file", args.inputFile, "does not exist.")
        exit()
    print("==========")
    print("Input File:", inputFile)
    print("==========")

    if numberOfFibers is not None:
        print("fibers to analyze per subject: ", numberOfFibers)
    else:
        print("fibers to analyze per subject: ALL")
    number_of_fibers = numberOfFibers
    fiber_length = fiberLength
    print("minimum length of fibers to analyze (in mm): ", fiber_length)
    points_per_fiber = numberOfFiberPoints
    print("Number of points in each fiber to process: ", points_per_fiber)

    # read data
    print("<wm_cluster_with_DEC.py> Reading input file:", inputFile)
    pd = wma.io.read_polydata(inputFile)

    if preproces:
        # preprocessing step: minimum length
        print("<wm_cluster_from_atlas.py> Preprocessing by length:", fiber_length, "mm.")
        pd2 = wma.filter.preprocess(pd, fiber_length, return_indices=False, preserve_point_data=True,
                                    preserve_cell_data=True, verbose=False)
    else:
        pd2 = pd

    # downsampling fibers if needed
    if number_of_fibers is not None:
        print("<wm_cluster_from_atlas.py> Downsampling to ", number_of_fibers, "fibers.")
        input_data = wma.filter.downsample(pd2, number_of_fibers, return_indices=False, preserve_point_data=True,
                                           preserve_cell_data=True, verbose=False)
    else:
        input_data = pd2

    fiber_array = fibers.FiberArray()
    fiber_array.convert_from_polydata(input_data, points_per_fiber=numberOfFiberPoints, data=data)
    feat = np.dstack((abs(fiber_array.fiber_array_r), fiber_array.fiber_array_a, fiber_array.fiber_array_s))
    return input_data, feat


def read_data(data_dir, numberOfFibers, fiberLength, numberOfFiberPoints, preproces=False, data='HCP'):
    inputDir = data_dir
    input_pd_fnames = wma.io.list_vtk_files(inputDir)
    num_pd = len(input_pd_fnames)

    input_pds = []
    x_arrays = []

    for i in range(num_pd):
        input_pd, x_array = convert_fiber_to_array(inputFile=input_pd_fnames[i],
                                                   numberOfFibers=numberOfFibers,
                                                   fiberLength=fiberLength,
                                                   numberOfFiberPoints=numberOfFiberPoints,
                                                   preproces=preproces,
                                                   data=data)

        input_pds.append(input_pd)
        x_arrays.append(x_array)


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


