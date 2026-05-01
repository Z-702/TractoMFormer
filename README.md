# TractoMFormer: MViT Classification with DTractoEmbedding Images

This repository provides the implementation of **TractoMFormer**, which performs subject-level classification based on DTractoEmbedding images generated from diffusion MRI tractography data.

The overall pipeline contains two main parts:

1. **Spectral Embedding / DTractoEmbedding Image Generation**
2. **MViT-based Classification with Multiple Input Modalities**

In addition, this repository also includes baseline models for comparison, including FC-1DCNN and ResNet-based classification methods.

---

## 1. Environment Setup

Build the environment following [INSTALL.md](https://github.com/Z-702/TractoMFormer_MVIT-classification/blob/main/TractoFormer-MVIT-main/INSTALL.md).

---

## 2. Spectral Embedding for DTractoEmbedding Image Generation

The first part of TractoMFormer is to generate DTractoEmbedding images from DTI inputs using the spectral embedding pipeline.

Use [embed.py](https://github.com/Z-702/TractoMFormer/tree/main/Spectral-embedding) to generate embedding images from DTI inputs.

The related codes are included in the [embed_tools](https://github.com/Z-702/TractoMFormer/tree/main/Spectral-embedding/embed_tools) folder.

Some codes were originally from the `DeepWMA` folder, which can be found in [DeepWMA](https://github.com/zhangfanmark/DeepWMA).

---

## 3. MViT-based Classification

The second part of TractoMFormer uses MViT to perform classification based on the generated DTractoEmbedding images.

Multiple input modalities, such as FA, MD, and density maps, are used to train and evaluate the model.

### 3.1 Dataset and Configuration Settings

For different datasets, change line 164 and line 169 in:

```bash
./mvit/dataset/tractoembedding.py
```

For model settings, edit the configuration file:

```bash
./configs/MVITv2_mri.yaml
```

### 3.2 Editing the Modalities Used

To change the input modalities, modify line 146 in:

```bash
./mvit/dataset/tractoembedding.py
```

For example:

```python
for mode in ['FA1', 'density', 'trace1']:
```

Then change the parameter `IN_CHANS` in:

```bash
./configs/MVITv2_mri.yaml
```

`IN_CHANS` should be set to the number of selected modalities.

For example, if three modalities are used, set:

```yaml
IN_CHANS: 3
```

### 3.3 Running the Code

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

### 3.4 CSV File Format

The CSV file should contain three columns:

```text
SUB_ID, DX_GROUP, fold
```

The program supports multi-fold evaluation. Five-fold cross-validation is recommended.

### 3.5 Parameter Description

`DATA_NUM` indicates the utilized embedding locations:

- `1`: left hemisphere
- `2`: left and right hemispheres
- `3`: left, right, and commissural streamlines

`DATA_AUG_NUM` indicates the number of augmented samples used.

### 3.6 Data Path Format

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

The resolution setting can be changed on line 64 of:

```bash
./mvit/dataset/tractoembedding.py
```

---

# Baseline Models

This repository also includes baseline methods for comparison with TractoMFormer.

## 1. FC-1DCNN Baseline

FC-1DCNN is a CNN-based baseline model that performs classification directly on raw DTI-derived tract features, such as FA, trace, and fiber count measurements.

Different from TractoMFormer, this method does not use DTractoEmbedding images. Instead, it uses numerical tract features as input and trains CNN-based classifiers for subject-level prediction.

### Key Features

- Uses raw DTI-derived tract features as input
- Supports feature selection, such as FA, trace, and fiber count
- Supports different hemisphere settings, including left hemisphere, right hemisphere, and commissural streamlines
- Supports several CNN-based architectures, such as 1D-CNN, 2D-CNN, LeNet, and 1.5D-CNN
- Supports 5-fold cross-validation
- Can be used for classification tasks such as sex classification or disease/control classification

### Quick Start

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

---

## 2. ResNet Baseline

The ResNet baseline performs classification using spectral embedding images. It uses a ResNet50 backbone pre-trained on ImageNet and replaces the final classifier head for the target classification task.

Compared with TractoMFormer, this baseline uses a conventional CNN backbone instead of MViT. It is used to evaluate whether transformer-based multi-scale modeling can improve classification performance on DTractoEmbedding images.

### Key Features

- Uses spectral embedding images as input
- Uses ResNet50 as the backbone network
- Supports transfer learning with ImageNet pre-trained weights
- Supports different modalities, such as FA, density, and trace
- Supports different image resolutions, such as 80, 160, and 320
- Supports 5-fold cross-validation
- Automatically saves model checkpoints during training

### Quick Start

```bash
python dataloader.py

CUDA_VISIBLE_DEVICES=1 nohup python run.py > train_FA1_160.log 2>&1 &
```
