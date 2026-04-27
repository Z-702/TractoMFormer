import argparse
import os
import nibabel as nib
import vtk
import numpy
import h5py
import glob
from nibabel.affines import apply_affine
import pandas

try:
    import whitematteranalysis as wma
except:
    print("Error importing white matter analysis package\n")
    raise

def main():
    #-----------------
    # Parse arguments
    #-----------------
    parser = argparse.ArgumentParser(
        description="",
        epilog="Written by Fan Zhang")
    
    parser.add_argument("-v", "--version",
        action="version", default=argparse.SUPPRESS,
        version='1.0',
        help="Show program's version number and exit")
    
    parser.add_argument(
        '-inputEmbed',
        help='Input npy.')
    parser.add_argument(
        '-inputMeasure',
        help='inpu npy.')
    parser.add_argument(
        '-outputdir',
        help='output img.')
    parser.add_argument(
        '-prefix', action="store", type=str, default='',
        help='output img.')
    parser.add_argument(
        '-ds', action="store", type=float, default='1.0',
        help='downsample rate')
    parser.add_argument(
        '-tractindexfile', action="store", type=str, default='',
        help='the tract each fiber blongs to.')

    args = parser.parse_args()

    outdir = os.path.join(args.outputdir, 'tractoembedding')
    os.makedirs(outdir, exist_ok=True)

    embedding = numpy.load(args.inputEmbed)
    diffusion_measures = numpy.load(args.inputMeasure)

    f = h5py.File(args.tractindexfile, "r")
    tract_list = f['tract_list'][:]
    tract_name = f['tract_name'][:]
    del f

    fiber_location = numpy.zeros(embedding.shape[0])
    for f_idx in range(fiber_location.shape[0]):
        t_idx = int(tract_list[f_idx])
        t_name = tract_name[t_idx]
        if "left" in str(t_name):
            fiber_location[f_idx] = 1
        elif "right" in str(t_name):
            fiber_location[f_idx] = 2
        elif "comm" in str(t_name):
            fiber_location[f_idx] = 0

    measures = ['FA1', 'FA2', 'trace1', 'trace2']
    Ds = [80, 160, 320, 640]

    N = 10
    # N = 100
    for ridx in range(0, N + 1):

        if ridx == 0: # full data
            f_rand_indices = numpy.arange(0, embedding.shape[0])
            ds_outdir = os.path.join(outdir, 'da-full')
        else: # ds data
            f_rand_indices = numpy.random.choice(embedding.shape[0], int(embedding.shape[0]*args.ds))
            f_rand_indices = numpy.sort(f_rand_indices)
            ds_outdir = os.path.join(outdir, 'da-%05d' % ridx)

        os.makedirs(ds_outdir, exist_ok=True)
        numpy.save(os.path.join(ds_outdir, 'fiber_indices'), f_rand_indices)

        print('da-%05d has %d fibers' % (ridx, f_rand_indices.shape[0]))

        for D in Ds:

            if D == 80:
                embedding_D = embedding * 12.5
            elif D == 160:
                embedding_D = embedding * 25
            elif D == 320:
                embedding_D = embedding * 50
            elif D == 640:
                embedding_D = embedding * 100
            elif D == 1280:
                embedding_D = embedding * 200

            print(embedding_D[:, 0].min(), embedding_D[:, 0].max())
            print(embedding_D[:, 1].min(), embedding_D[:, 1].max())

            # diffusion measures
            for midx, measure in enumerate(measures):
                print(midx, measure)

                embedding_img = numpy.zeros((D, D, 3*4)) # Three locations x four measures [mean_C, mean_L, mean_R, min_C, min_L, min_R, max_C, max_L, max_R, median_C, median_L, median_R, ]
                embedding_img_count = numpy.zeros((D, D, 3))

                for frindx, fidx in enumerate(f_rand_indices):
                    dx = embedding_D[fidx, 0].round() + D / 2 - D / 10
                    dy = embedding_D[fidx, 1].round() + D / 2 - D / 10
                    # dz = embedding_D[fidx, 2].round() + D / 2 - D / 10

                    dx = dx.astype(int)
                    dy = dy.astype(int)
                    # dz = dz.astype(int)

                    location = fiber_location[fidx].astype(int)
                    CLR_slices = numpy.array(range(0, 12, 3)) + location

                    if embedding_img_count[dx, dy, location] == 0:
                        embedding_img[dx, dy, CLR_slices] = diffusion_measures[fidx, midx, :].squeeze()

                        embedding_img_count[dx, dy, location] += 1
                    else:
                        # compute mean
                        tmp = embedding_img[dx, dy, CLR_slices]
                        tmp = tmp * embedding_img_count[dx, dy, location] + diffusion_measures[fidx, midx, :].squeeze()
                        tmp = tmp / (embedding_img_count[dx, dy, location] + 1)
                        embedding_img[dx, dy, CLR_slices] = tmp

                        embedding_img_count[dx, dy, location] += 1

                    if frindx % 10000 == 0:
                        print("Fiber: %s, location: %s, Fiber measure: %s" % \
                              (frindx, location, diffusion_measures[fidx, midx, :]))

                output_fname = os.path.join(ds_outdir, args.prefix+'-'+measure+'_CLR_sz'+str(D)+'.nii.gz')
                if not os.path.exists(output_fname):
                    print('Saving:',output_fname)
                    img = nib.Nifti1Image(embedding_img[:, :, :3], numpy.eye(4))
                    nib.save(img, output_fname)

                output_fname = os.path.join(ds_outdir, args.prefix+'-density_CLR_sz'+str(D)+'.nii.gz')
                if not os.path.exists(output_fname):
                    print('Saving:',output_fname)
                    img = nib.Nifti1Image(embedding_img_count, numpy.eye(4))
                    nib.save(img, output_fname)

if __name__ == '__main__':
    main()


# 1280 x 1280: 640
# 640 x 640: 320 
# 320 x 320: 160




