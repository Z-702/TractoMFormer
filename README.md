## MViT Classification with DGCNN-based Embedding Images

This code uses MViT for DGCNN-based embedding image classification.

Build the environment following [INSTALL.md](https://github.com/Z-702/TractoMFormer_MVIT-classification/blob/main/TractoFormer-MVIT-main/INSTALL.md).

## Using Spectral Embedding to Generate DTractoEmbedding Images

Use [embed.py](https://github.com/Z-702/TractoMFormer/tree/main/Spectral-embedding) to generate embedding images from DTI inputs.

The related codes are included in the [embed_tools](https://github.com/Z-702/TractoMFormer/tree/main/Spectral-embedding/embed_tools) folder.

Some codes were originally from the `DeepWMA` folder, which can be found on each server. Therefore, there is no need to include them in this repository.

## Using Multiple Inputs for Model Training and Evaluation

Multiple inputs, such as FA, MD, and density maps, are used to train and evaluate the model.

1. For different datasets, change line 164 and line 169 in `./mvit/dataset/tractoembedding.py`.
2. For model settings, edit the configuration file `./configs/MVITv2_mri.yaml`.

## Editing the Modalities Used

1. Change line 146 in `./mvit/dataset/tractoembedding.py`.

```python
for mode in ['FA1', 'density', 'trace1']:
```

2. Change the parameter `IN_CHANS` in `./configs/MVITv2_mri.yaml`.

`IN_CHANS` should be set to the number of selected modalities.

## Running the Code

Execute the code as follows:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$(pwd) \
nohup python ./tools/main.py \
    --cfg configs/MVITv2_mri.yaml \
    DATA.PATH_TO_DATA_DIR fold.csv \
    NUM_GPUS 2 \
    TRAIN.BATCH_SIZE 10 \
    OUTPUT_DIR path/to/your/model \
    DATA_NUM {num} \
    DATA_AUG_NUM {N} \
    > output_new500_v2.log 2>&1 &
```

## CSV File Format

The CSV file should contain three columns:

```text
SUB_ID, DX_GROUP, fold
```

The program supports multi-fold evaluation. Five-fold cross-validation is recommended.

## Parameter Description

`DATA_NUM` indicates the utilized embedding locations:

- `1`: left hemisphere
- `2`: left and right hemispheres
- `3`: left, right, and commissural streamlines

`DATA_AUG_NUM` indicates the number of augmented samples used.

## Data Path Format

The data path should follow this format:

```bash
/data01/zixi/HCP_500_vtk/129634/tractoembedding/da-full/129634-trace1_CLR_sz640.nii.gz
```

The path components are defined as follows:

1. `/data01/zixi/HCP_500_vtk/129634/tractoembedding`: root path
2. `129634`: subject ID
3. `da-full`: augmentation index
4. `trace1`: modality
5. `sz640`: resolution

By default, three resolutions are used:

```python
['sz80', 'sz160', 'sz320']
```

The resolution setting can be changed on line 64 of `./mvit/dataset/tractoembedding.py`.

## Baseline Models

### FC-1DCNN - CNN-based Classification on Raw DTI Features

Predicts subject age/sex directly from diffusion measurements (FA, trace, fiber counts) using ensemble CNNs.

**Key Features:**
- Ensemble learning: one model per input feature
- Multiple architectures: 1D-CNN, 2D-CNN, LeNet, 1.5D-CNN
- 5-fold cross-validation with real-time Visdom visualization
- Supports flexible feature selection and hemisphere combination

**Quick Start:**
```bash
cd FC-1DCNN/dti
nohup python main.py \
  --data-path /data05/learn2reg/zixi/csv_CNP_150 \
  --demographics-csv /data05/learn2reg/CNP_150_clean.csv \
  --subject-col SUB_ID \
  --dx-col DX_GROUP \
  --MODEL 1D-CNN \
  --INPUT-FEATURES trace1.Mean trace2.Mean \
  --HEMISPHERES left-hemisphere right-hemisphere commissural \
  --target sex \
  --NUM_CLASSES 2 \
  --LOSS CE \
  --epochs 200 \
  --L2 1e-4 \
  > train_cnp_trace.log 2>&1 &
```

### ResNet - Transfer Learning on Spectral Embedding Images
Uses ResNet50 (ImageNet pre-trained) to classify embedding images for binary classification (PD vs Control).
**Key Features:**
- Pre-trained backbone with custom classifier head
- Supports multiple modalities (FA, density, trace) and resolutions (80, 160, 320)
- 5-fold cross-validation with automatic model checkpointing
- Flexible CSV-based fold assignmen

**Quick Start:**
```bash
python dataloader.py
CUDA_VISIBLE_DEVICES=1 nohup python run.py > train_FA1_160.log 2>&1 &
```
