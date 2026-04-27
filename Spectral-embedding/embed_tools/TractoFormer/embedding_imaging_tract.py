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

    print(tract_name)
    fiber_location = numpy.zeros(embedding.shape[0])
    for f_idx in range(fiber_location.shape[0]):
        t_idx = int(tract_list[f_idx])
        fiber_location[f_idx] = t_idx

    tract_location = numpy.zeros(len(tract_name))
    for tr_idx in range(len(tract_name)):
        t_name = tract_name[tr_idx]
        if "left" in str(t_name):
            tract_location[tr_idx] = 1
        elif "right" in str(t_name):
            tract_location[tr_idx] = 2
        elif "comm" in str(t_name):
            tract_location[tr_idx] = 0

    measures = ['tract']
    Ds = [80, 160, 320, 640]

    N = 0
    for ridx in range(N + 1):

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

                embedding_img = numpy.zeros((D, D, len(tract_name))) # Three locations x four measures [mean_C, mean_L, mean_R, min_C, min_L, min_R, max_C, max_L, max_R, median_C, median_L, median_R, ]
                embedding_img_count = numpy.zeros((D, D, 3))

                print(embedding_img.shape)
                for frindx, fidx in enumerate(f_rand_indices):
                    dx = embedding_D[fidx, 0].round() + D / 2 - D / 10
                    dy = embedding_D[fidx, 1].round() + D / 2 - D / 10
                    # dz = embedding_D[fidx, 2].round() + D / 2 - D / 10

                    dx = dx.astype(int)
                    dy = dy.astype(int)
                    # dz = dz.astype(int)

                    location = fiber_location[fidx].astype(int)
                    embedding_img[dx, dy, location] +=1

                    if frindx % 10000 == 0:
                        print("Fiber: %s, location: %s, Fiber measure: %s" % \
                              (frindx, location, diffusion_measures[fidx, midx, :]))

                embedding_img_c = embedding_img[:, :, tract_location == 0]
                embedding_img_c_map = numpy.argmax(embedding_img_c, axis=2)
                embedding_img_l = embedding_img[:, :, tract_location == 1]
                embedding_img_l_map = numpy.argmax(embedding_img_l, axis=2)
                embedding_img_r = embedding_img[:, :, tract_location == 2]
                embedding_img_r_map = numpy.argmax(embedding_img_r, axis=2)

                embedding_img_map = numpy.stack((embedding_img_c_map, embedding_img_l_map, embedding_img_r_map)).transpose([1,2,0])
                embedding_img_map = embedding_img_map.astype(float)
                output_fname = os.path.join(ds_outdir, args.prefix+'-'+measure+'_sz'+str(D)+'.nii.gz')
                if not os.path.exists(output_fname):
                    print('Saving:',output_fname)
                    img = nib.Nifti1Image(embedding_img[:, :, :], numpy.eye(4))
                    nib.save(img, output_fname)

                output_fname = os.path.join(ds_outdir, args.prefix+'-'+measure+'_map_sz'+str(D)+'.nii.gz')
                if not os.path.exists(output_fname):
                    print('Saving:',output_fname)
                    img = nib.Nifti1Image(embedding_img_map[:, :, :], numpy.eye(4))
                    nib.save(img, output_fname)


if __name__ == '__main__':
    main()


# 1280 x 1280: 640
# 640 x 640: 320 
# 320 x 320: 160



    # /home/fan/Software/miniconda3/py310_23.3.1-0/envs/tractoformer/bin/python \
    #         /home/fan/Projects/TractoFormer/embedding_imaging_tract.py \
    #         -inputEmbed /data01/embed-tmp/301/tractoembedding/sub-13000/embed.npy \
    #         -inputMeasure /data01/embed-tmp/301/tractoembedding/sub-13000/embed_feat.npy \
    #         -outputdir /data01/embed-tmp/301/tractoembedding/sub-13000/ \
    #         -prefix sub-13000 \
    #         -tractindexfile /data01/embed-tmp/301/tractoembedding/sub-13000/tracts/sub-13000.h5