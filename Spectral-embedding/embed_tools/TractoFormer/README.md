# TractoFormer
TractoFormer

# step 1: preprocessing

    wm_register to atlas:
    wm_preprocessing_all -l 40

# step 2: DeepWMA and fiber feature extracion
    
    /home/fan/Software/miniconda3/py310_23.3.1-0/envs/DeepWMA/bin/python \
            /home/fan/Projects/DeepWMA/dlt_extract_tract_feat.py \
            /data01/embed-tmp/301/data-dsl40/sub-13001_ses-1_run-1_tractography_tfm_pp.vtp \
            /data01/embed-tmp/301/tractoembedding/sub-13001/deepwma \
            -outPrefix sub-13001 -feature RAS-3D -numPoints 15
    
    /home/fan/Software/miniconda3/py310_23.3.1-0/envs/DeepWMA/bin/python \
            /home/fan/Projects/DeepWMA/dlt_test.py \
            /home/fan/Projects/DeepWMA/SegModels/CNN/cnn_model.h5 \
            -modelLabelName /home/fan/Projects/DeepWMA/SegModels/CNN/cnn_label_names.h5 \
            /data01/embed-tmp/301/tractoembedding/sub-13001/deepwma/sub-13001_featMatrix.h5 \
            /data01/embed-tmp/301/tractoembedding/sub-13001/deepwma/ \
            -outPrefix sub-13001 \
            -tractVTKfile /data01/embed-tmp/301/data-dsl40/sub-13001_ses-1_run-1_tractography_tfm_pp.vtp  

    /home/fan/Software/miniconda3/py310_23.3.1-0/envs/tractoformer/bin/python \
            /home/fan/Projects/TractoFormer/assess_cluster_location_by_hemisphere.py \
            /data01/embed-tmp/301/tractoembedding/sub-13001/deepwma/sub-13001_prediction_tracts_outlier_removed/ \
            -clusterLocationFile /home/fan/Projects/TractoFormer/deepwma-hemi.csv \
            -outputDirectory /data01/embed-tmp/301/tractoembedding/sub-13001/deepwma/sub-13001_separated/ 

    /home/fan/Software/miniconda3/py310_23.3.1-0/envs/tractoformer/bin/python \
            /home/fan/Projects/TractoFormer/append_tracts.py \
            /data01/embed-tmp/301/tractoembedding/sub-13001/deepwma/sub-13001_separated/ \
            /data01/embed-tmp/301/tractoembedding/sub-13001/deepwma/ \
            -appendedTractName sub-13001 

# step 3: generate embedding

    /home/fan/Software/miniconda3/py310_23.3.1-0/envs/tractoformer/bin/python \
            /home/fan/Projects/Tractofomer-Optimization/OPT/main.py \
            -intract /data01/embed-tmp/301/tractoembedding/sub-13001/deepwma/sub-13001.vtp \
            -modelpath /home/fan/Projects/Tractofomer-Optimization/OPT/weights/Atlas-250.pt \
            -mode test -outdir /data01/embed-tmp/301/tractoembedding/sub-13001 
    
# step 4: extract diffusion feature

    /home/fan/Software/miniconda3/py310_23.3.1-0/envs/tractoformer/bin/python \
            /home/fan/Projects/TractoFormer/extract_measures_tensors.py \
            -inputVTK /data01/embed-tmp/301/tractoembedding/sub-13001/deepwma/sub-13001.vtp \
            -outputNPY /data01/embed-tmp/301/tractoembedding/sub-13001/embed_feat.npy
            
# step 5 tractoembedding 
    
    /home/fan/Software/miniconda3/py310_23.3.1-0/envs/tractoformer/bin/python \
            /home/fan/Projects/TractoFormer/embedding_imaging.py \
            -inputEmbed /data01/embed-tmp/301/tractoembedding/sub-13001/embed.npy \
            -inputMeasure /data01/embed-tmp/301/tractoembedding/sub-13001/embed_feat.npy \
            -outputdir /data01/embed-tmp/301/tractoembedding/sub-13001/ \
            -prefix sub-100301 \
            -D 160 \
            -tractindexfile /data01/embed-tmp/301/tractoembedding/sub-13001/deepwma/sub-13001.h5
    
