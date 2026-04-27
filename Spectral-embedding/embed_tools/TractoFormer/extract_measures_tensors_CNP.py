import argparse
import os
import vtk
import numpy
import whitematteranalysis as wma
from numpy import linalg as LA

def main():
    #-----------------
    # Parse arguments
    #-----------------
    parser = argparse.ArgumentParser(
        description="Convert a fiber tract or cluster (vtk) to a voxel-wise fiber density image (nii.gz). ",
        epilog="Written by Fan Zhang")
    
    parser.add_argument("-v", "--version",
        action="version", default=argparse.SUPPRESS,
        version='1.0',
        help="Show program's version number and exit")
    
    parser.add_argument(
        '-inputVTK',
        help='Input VTK/VTP.')
    parser.add_argument(
        '-outputNPY',
        help='Output Values.')
    
    args = parser.parse_args()

    # args.inputVTK = "/data01/embed-tmp/301/data-dsl40/sub-13001_ses-1_run-1_tractography_tfm_pp.vtp"
    # args.outputNPY = "/data01/embed-tmp/301/tractoembedding/sub-13001/measure.npy"

    print('Reading tract data:', args.inputVTK)
    inpd = wma.io.read_polydata(args.inputVTK)

    inpoints = inpd.GetPoints()
    inpointdata = inpd.GetPointData()

    # Diffusion measures
    measures = ['FA1', 'FA2', 'trace1', 'trace2']
    diffusion_measures = numpy.zeros((inpd.GetNumberOfLines(), len(measures), 4)) # Mean, Min, Max, Median
    
    # TODO: FS regions

    output_diffusion_measures_npy = args.outputNPY

    inpd.GetLines().InitTraversal()
    for lidx in range(0, inpd.GetNumberOfLines()):

        ptids = vtk.vtkIdList()
        inpd.GetLines().GetNextCell(ptids)

        if lidx % 1000 == 0:
            print('Line', lidx, ' - ', ptids.GetNumberOfIds())


        # Get tensor and compute FA, Trace
        fa1_list = []
        fa2_list = []
        trace1_list = []
        trace2_list = []
        for pidx in range(0, ptids.GetNumberOfIds()):

            array = inpointdata.GetArray("tensor1")
            val = array.GetTuple(ptids.GetId(pidx))
            val_mat = numpy.array([[val[0], val[1], val[2]], [val[3], val[4], val[5]], [val[6], val[7], val[8]]])
            evals, _ = LA.eig(val_mat)

            # Make sure not to get nans
            all_zero = (evals == 0).all(axis=0)
            ev1, ev2, ev3 = evals
            fa1 = numpy.sqrt(0.5 * ((ev1 - ev2) ** 2 +
                                (ev2 - ev3) ** 2 +
                                (ev3 - ev1) ** 2) /
                         ((evals * evals).sum(0) + all_zero))

            trace1 = evals.sum(0)


            array = inpointdata.GetArray("tensor2")
            val = array.GetTuple(ptids.GetId(pidx))
            val_mat = numpy.array([[val[0], val[1], val[2]], [val[3], val[4], val[5]], [val[6], val[7], val[8]]])
            evals, _ = LA.eig(val_mat)

            # Make sure not to get nans
            all_zero = (evals == 0).all(axis=0)
            ev1, ev2, ev3 = evals
            fa2 = numpy.sqrt(0.5 * ((ev1 - ev2) ** 2 +
                                (ev2 - ev3) ** 2 +
                                (ev3 - ev1) ** 2) /
                         ((evals * evals).sum(0) + all_zero))

            trace2 = evals.sum(0)

            fa1_list.append(fa1)
            fa2_list.append(fa2)
            trace1_list.append(trace1)
            trace2_list.append(trace2)

        fa1_list = numpy.array(fa1_list)
        fa2_list = numpy.array(fa2_list)
        trace1_list = numpy.array(trace1_list)
        trace2_list = numpy.array(trace2_list)

        diffusion_measures[lidx, 0, 0] = fa1_list.mean()
        diffusion_measures[lidx, 0, 1] = fa1_list.min()
        diffusion_measures[lidx, 0, 2] = fa1_list.max()
        diffusion_measures[lidx, 0, 3] = numpy.median(fa1_list)

        diffusion_measures[lidx, 1, 0] = fa2_list.mean()
        diffusion_measures[lidx, 1, 1] = fa2_list.min()
        diffusion_measures[lidx, 1, 2] = fa2_list.max()
        diffusion_measures[lidx, 1, 3] = numpy.median(fa2_list)

        diffusion_measures[lidx, 2, 0] = trace1_list.mean()
        diffusion_measures[lidx, 2, 1] = trace1_list.min()
        diffusion_measures[lidx, 2, 2] = trace1_list.max()
        diffusion_measures[lidx, 2, 3] = numpy.median(trace1_list)

        diffusion_measures[lidx, 3, 0] = trace2_list.mean()
        diffusion_measures[lidx, 3, 1] = trace2_list.min()
        diffusion_measures[lidx, 3, 2] = trace2_list.max()
        diffusion_measures[lidx, 3, 3] = numpy.median(trace2_list)

        
        # # The diffsuion of each fiber: mean, min, max and mediam
        # if not os.path.exists(output_diffusion_measures_npy):
        #     for midx, measure in enumerate(measures):
        #
        #         tmp_list = []
        #         for pidx in range(0, ptids.GetNumberOfIds()):
        #
        #             array = inpointdata.GetArray(measure)
        #             val = array.GetTuple(ptids.GetId(pidx))
        #             tmp_list.append(val[0])
        #         tmp_list = numpy.array(tmp_list)
        #
        #         diffusion_measures[lidx, midx, 0] = tmp_list.mean()
        #         diffusion_measures[lidx, midx, 1] = tmp_list.min()
        #         diffusion_measures[lidx, midx, 2] = tmp_list.max()
        #         diffusion_measures[lidx, midx, 3] = numpy.median(tmp_list)

        # TODO: The FS region eah fiber passes
        # if not os.path.exists(output_freesurfer_npy):
        #     print("TODO")

    # # testing only to make sure locations are correct.
    # pd_c_list = wma.cluster.mask_all_clusters(inpd, cluster_location, 3, preserve_point_data=False, preserve_cell_data=False,verbose=True)
    # for idx, pd_c in enumerate(pd_c_list):
    #     print("saving to: ", args.outputNPY.replace('.npy', '-'+str(idx)+'.vtp'))
    #     wma.io.write_polydata(pd_c, args.outputNPY.replace('.npy', '-'+str(idx)+'.vtp'))

    if not os.path.exists(output_diffusion_measures_npy):
        numpy.save(output_diffusion_measures_npy, diffusion_measures)
        print("saving to: ", output_diffusion_measures_npy)


if __name__ == '__main__':
    main()





