# Tractofomer-Optimization

## 配置环境

```bash

conda create --name tractoformer python=3.10.9
conda activate tractoformer

git clone https://github.com/yPanStupidog/Tractofomer-Optimization.git
cd Tractofomer-Optimization
pip install git+https://github.com/SlicerDMRI/whitematteranalysis.git@0d11e46775fce0001a859f10037dffe26f14985b
pip3 install -r requirements.txt

python main.py -intract /data01/embed-tmp/301/data-dsl40/sub-13001_ses-1_run-1_tractography_tfm_pp.vtp \
               -modelpath weights/Atlas-250.pt -mode test \
               -outdir /data01/embed-tmp/301/tractoembedding/sub-13001
```


