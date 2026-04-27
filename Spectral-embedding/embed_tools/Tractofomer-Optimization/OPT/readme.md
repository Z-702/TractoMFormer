# 训练模型
下载release中的**clustered_whole_brain.vtp**、**embed.npy**下载到 **./datasets**路径下。如果 **./weights**下没有文件，请设置--modeldir为None。
```bash
CUDA_VISIBLE_DEVICES=<CUDA ids> python main.py --modeldir None \
    --save None \
    --mode train
```
默认使用100W fibers的Atlas data做训练。

# 测试模型
下载release中的**Atlas-250.pt**到 **./weights**路径下。如果不需要保存生成的embedding，需要设置--save为False，默认为True。
```bash
CUDA_VISIBLE_DEVICES=<CUDA ids> python main.py --save False
```
