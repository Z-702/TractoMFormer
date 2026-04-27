
datafolder=/data01/embed-tmp/301/data-dsl40/
outdir=/data01/embed-tmp/301/tractoembedding
mkdir -p $outdir

while read subID
do
  echo $subID
  tractvtk=$(ls $datafolder/$subID*ses-1_run-1*vtp)
  echo $tractvtk

  embeddingfolder=$outdir/$subID
  mkdir -p $embeddingfolder

  # deepwma
  embedtract=$embeddingfolder/tracts/${subID}.vtp
  embedindex=$embeddingfolder/tracts/${subID}.h5
  if [ ! -f $embedtract ]; then
    $2 /home/fan/Software/miniconda3/py310_23.3.1-0/envs/DeepWMA/bin/python \
        /home/fan/Projects/DeepWMA/dlt_extract_tract_feat.py \
        $tractvtk $embeddingfolder/tracts \
        -outPrefix $subID -feature RAS-3D -numPoints 15

    $2 /home/fan/Software/miniconda3/py310_23.3.1-0/envs/DeepWMA/bin/python \
            /home/fan/Projects/DeepWMA/dlt_test.py \
            /home/fan/Projects/DeepWMA/SegModels/CNN/cnn_model.h5 \
            -modelLabelName /home/fan/Projects/DeepWMA/SegModels/CNN/cnn_label_names.h5 \
            $embeddingfolder/tracts/${subID}_featMatrix.h5 \
            $embeddingfolder/tracts/ \
            -outPrefix $subID \
            -tractVTKfile $tractvtk

    $2 /home/fan/Software/miniconda3/py310_23.3.1-0/envs/tractoformer/bin/python \
            /home/fan/Projects/TractoFormer/assess_cluster_location_by_hemisphere.py \
            $embeddingfolder/tracts/${subID}_prediction_tracts_outlier_removed/ \
            -clusterLocationFile /home/fan/Projects/TractoFormer/deepwma-hemi.csv \
            -outputDirectory $embeddingfolder/tracts/${subID}_separated/

    $2 /home/fan/Software/miniconda3/py310_23.3.1-0/envs/tractoformer/bin/python \
            /home/fan/Projects/TractoFormer/append_tracts.py \
            $embeddingfolder/tracts/${subID}_separated/ \
            $embeddingfolder/tracts/ \
            -appendedTractName $subID

    $2 rm $embeddingfolder/tracts/${subID}_featMatrix.h5
    $2 rm -r $embeddingfolder/tracts/${subID}_prediction_tracts_outlier_removed/
    $2 rm -r $embeddingfolder/tracts/${subID}_separated/
    $2 rm $embeddingfolder/tracts/${subID}_multi_tract_specific_prediction_mask.h5
  fi

  embednpy=$embeddingfolder/embed.npy
  if [ ! -f $embednpy ];then
    $2 /home/fan/Software/miniconda3/py310_23.3.1-0/envs/tractoformer/bin/python \
            /home/fan/Projects/Tractofomer-Optimization/OPT/main.py \
            -intract $embedtract \
            -modelpath /home/fan/Projects/Tractofomer-Optimization/OPT/weights/Atlas-250.pt \
            -mode test -outdir $embeddingfolder
  fi

  embeddiffusionmeasure=$embeddingfolder/embed_feat.npy
  if [ ! -f $embeddiffusionmeasure ]; then
      $2 /home/fan/Software/miniconda3/py310_23.3.1-0/envs/tractoformer/bin/python \
            /home/fan/Projects/TractoFormer/extract_measures_tensors.py \
            -inputVTK $embedtract \
            -outputNPY $embeddiffusionmeasure
  fi

  if [ ! -f $embeddingfolder/tractoembedding/da-00100/fiber_indices.npy ]; then
      $2 /home/fan/Software/miniconda3/py310_23.3.1-0/envs/tractoformer/bin/python \
            /home/fan/Projects/TractoFormer/embedding_imaging.py \
            -inputEmbed $embednpy \
            -inputMeasure $embeddiffusionmeasure \
            -outputdir $embeddingfolder \
            -prefix $subID \
            -ds 0.8 \
            -tractindexfile $embedindex
  fi

done < /data01/embed-tmp/301/$1