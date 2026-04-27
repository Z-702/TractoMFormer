# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import json
import pandas as pd
import os
from itertools import product
import random
import re
import numpy as np

import mvit.utils.logging as logging
import torch
import torch.utils.data
from mvit.utils.env import pathmgr
from PIL import Image
from torchvision import transforms as transforms_tv
from pathlib import Path
import nibabel as nib
from mvit.datasets.rand_augment import rand_augment_transform
from mvit.datasets.random_erasing import RandomErasing
from mvit.datasets.normalization import Normalize


from .build import DATASET_REGISTRY
# from .transform import transforms_imagenet_train

logger = logging.get_logger(__name__)
# data_5_fold = np.array([[  9, 286, 396, 264, 304, 409,  17,  34,  35, 374, 349,  36, 160,
#         187, 161, 324, 201, 416, 198,   8,  44, 251, 142, 372, 211,  57,
#         314, 147,  14, 194, 332, 323, 377, 226, 140, 176, 375, 361, 162,
#         267,  12,  33, 365, 205, 364,  48, 370, 301,  96,  79, 217,  88,
#         335,  39,  78,  55, 227, 110, 223,  77, 293, 401,  51, 441, 337,
#          26, 333, 325, 326, 197,  64,  98, 256, 350, 329, 167, 424,  76,
#         221, 299, 392,  67,  29, 430, 138,  97, 219, 384, 311],
#        [ 53, 358,  50, 297, 391, 313,  86, 355, 183, 216, 433, 144, 345,
#         257,  72, 371, 425, 177, 263, 240, 419, 320, 103, 266, 334,  20,
#         290, 106, 202, 169, 243, 369, 208, 118, 229,  38, 181, 121, 260,
#         209, 265, 390, 154, 414, 128, 236, 179, 120, 259, 269, 125, 126,
#         341, 403, 379, 353,  13,  68, 252, 283,  71, 101,  10, 343, 357,
#         133, 359, 356, 360, 373,   0, 164, 212, 386, 174, 191, 163, 300,
#         270, 413, 406, 317, 237, 420, 173, 385,  56,  62, 233],
#        [368,  42,  59, 220, 411,  89, 319, 432, 152, 305, 347, 376, 302,
#         435, 367, 165,  16, 381,  63,  37, 316, 378,  85,  24, 387, 402,
#         204, 249, 254,  69, 278, 363, 436, 171, 184,  25, 280,   5,  73,
#         253, 104,  40,  52, 285, 284, 268, 178, 258,  65, 279,  49, 158,
#         157, 330, 123, 159, 291, 428, 442, 346, 107, 119, 117, 399,  80,
#         427, 327, 440, 303,  58, 444, 156, 109, 190, 207, 214,  74,  23,
#         307, 175, 225, 380,  31, 248, 338, 168,  91, 289, 238],
#        [186, 410, 407, 145,  75, 235, 132,  70,  83, 423, 153, 348,  46,
#         115, 105, 336, 113, 130, 395, 292,  11, 195, 388, 155, 255,  30,
#         134, 232, 135, 429, 262, 277,   3, 136, 273, 149, 166, 405, 137,
#         250, 418, 231, 281, 434, 203, 218, 274, 404, 443, 228, 182, 122,
#         131, 193, 351, 100, 172,  21,  81, 148, 342, 321,   7,  32, 200,
#         394, 383, 224, 328, 150, 139,  90, 124, 309, 230, 127,  41, 331,
#          54, 241, 116, 185, 215, 421, 244, 315,  18, 246, 362],
#        [ 47, 141, 210, 189,  93, 340,  61, 222, 308, 247,  15, 306,  84,
#         366,  43,  94, 294, 213, 196,  87, 242, 288, 129, 352,  66,  82,
#         287, 143, 310, 275, 245, 102, 426, 389,   2, 344, 276, 439, 282,
#         438, 188,  99, 296,  92, 437,  45, 431, 271, 170, 272, 382, 417,
#         114, 400,  19, 408, 393, 112, 422, 192, 397, 261,   6, 412, 180,
#           1,  22, 318, 339, 108,   4, 111, 295,  60, 298,  27, 398, 206,
#         199, 239,  28, 234, 322, 312, 151, 354, 415,  95, 146]])
precison_list = ['sz80','sz160','sz320',]#这个大小不能改变，要不然会爆显存
precison_list.reverse()
# Tractoembedding_read_counter = 0 #用来进行计数的计数器

@DATASET_REGISTRY.register()

class Tractoembedding(torch.utils.data.Dataset):
    """tractoembedding dataset."""

    def __init__(self, cfg, mode, n_fold=0,num_retries=10):
        self.num_retries = num_retries
        self.cfg = cfg
        self.mode = mode
        self.data_path = cfg.DATA.PATH_TO_DATA_DIR #阅读传入的csv文件
        self.return_subid=False
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for tractoembedding".format(mode)
        if mode == 'train':
            # train_fold = np.concatenate(data_5_fold[mask])
            # self.data = (pd.read_csv(self.data_path).to_numpy())[train_fold,:]
            demo = pd.read_csv(self.data_path)
            demo = demo[demo['fold']!=n_fold]
            print(demo.shape)
            self.data = demo[['SUB_ID','DX_GROUP']].to_numpy()
            # self.data = pd.read_csv(self.data_path).loc 
        elif mode == 'val':
            demo = pd.read_csv(self.data_path)
            demo = demo[demo['fold']==n_fold]
            print(demo.shape)
            self.data = demo[['SUB_ID','DX_GROUP']].to_numpy()
        elif mode == 'test':
            demo = pd.read_csv(self.data_path)
            if 'fold' in demo.columns:
                demo = demo[demo['fold'] == n_fold]
            else:
                print(f"[test] No 'fold' column found. Using all {len(demo)} samples.")
            print(f"[test] Loaded {len(demo)} samples from {self.data_path}")
            self.data = demo[['SUB_ID', 'DX_GROUP']].to_numpy()
    def _prepare_im(self, im):
        train_size, test_size = (
            self.cfg.DATA.TRAIN_CROP_SIZE,
            self.cfg.DATA.TEST_CROP_SIZE,
        )
        mean = self.cfg.DATA.MEAN
        std = self.cfg.DATA.STD
        if self.mode == "train":
            t = []
            t.append(transforms_tv.ToTensor())
            t.append(transforms_tv.ConvertImageDtype(torch.float32,))
            # t.append(Normalize(mean,std))
            # t.append(transforms_tv.RandomErasing(self.cfg.AUG.RE_PROB,))
        else:
            t = []
            t.append(transforms_tv.ToTensor())
            t.append(transforms_tv.ConvertImageDtype(torch.float32,))
            # t.append(Normalize(mean,std))
            # t.append(transforms_tv.Normalize(self.cfg.DATA.MEAN, self.cfg.DATA.STD))

        aug_transform = transforms_tv.Compose(t)
        im = aug_transform(im)
        
        return im
    @torch.jit.ignore
  

    def _load_data(self, data_path,precision,sub_info=None,aug_num=None,):
        global Tractoembedding_read_counter  # 声明使用外部变量
        if aug_num is None:
            data_path = data_path/f'da-full'
            # data_list = sorted(data_path/(f'da-full/sub-{sub_info}-{mode}_CLR_{precision}.nii.gz'))
        else:
            assert isinstance(aug_num,int)
            #aug_num = 100-aug_num 
            data_path = data_path/f'da-{aug_num+1:05d}' #这里进行了修改，确保是从da-0001到da-0010
        data_list = []
        for mode in ['FA1','density','trace1']:  # 这里不用修改，直接运行FA1, density,trace1就行
            path = data_path/f'{sub_info}-{mode}_CLR_{precision}.nii.gz'
            data_list.append(path)
            # Tractoembedding_read_counter += 1
            # if self.mode=="val":
            #     logger.info(f"[{Tractoembedding_read_counter}] [LOAD] Reading file: {str(path)}")  # 写入日志
                # data_list = sorted(data_path.glob(f'*{mode}*{precision}*RS{aug_num:04d}.nii.gz'))
        assert len(data_list) == 3 ,f'only {len(data_list)} {precision} data for {data_path}'  #目前支持只使用1个模态
        #assert len(data_list) == 1 ,f'only {len(data_list)} {precision} data for {data_path}'
        load_func = lambda x: nib.load(x).get_fdata()[...,:3] if x.name.endswith('.nii.gz') else np.load(x)
        
        data = np.nan_to_num(np.stack(list(map(load_func,data_list)),-2),0)
        return data
    def __load__(self, index,aug_num):
        try:
            # Load the image
            sub_info = self.data[index,0]
            if self.cfg.DATA_AUG_NUM == 1:#如果DATA_AUG_NUM为1，则不需要进行数据增强,输入的是da_full
                data_path = Path('/data01/zixi/tractoembedding_PPMI_V2')/f'{sub_info}/tractoembedding'#这里的地址需要更改为数据的地址
                aug_num = None
                
            else:
                data_path = Path('/data01/zixi/tractoembedding_PPMI_V2')/f'{sub_info}/tractoembedding'#这里的操作同上
            data_dict = {precison: self._load_data(data_path,precison,sub_info,aug_num,) for precison in precison_list}

            data = []
            if self.cfg.DATA_MODE == 'fusion':

                for i in range(self.cfg.DATA_NUM):
                    #data.append([self._prepare_im(data_dict[precision][...,i]) for precision in precison_list])
                    # ✅ 修复后的拼接逻辑（只拼 FA1）
                    data.append([self._prepare_im(data_dict[precision][..., 0]) for precision in precison_list])

            return data

        except Exception as e:
            print(e)
            return None

    def __getitem__(self, index):

        if self.mode == 'train':
            sub_index, aug_num = index // self.cfg.DATA_AUG_NUM, index % self.cfg.DATA_AUG_NUM
            label = self.data[sub_index, 1] - 1
            im = self.__load__(sub_index, aug_num)
            for _ in range(self.num_retries):
                if im is None:
                    aug_num = random.randint(0, self.cfg.DATA_AUG_NUM - 1)
                    im = self.__load__(sub_index, aug_num)
                else:
                    break
            sub_id = self.data[sub_index, 0]   # ✅ 训练模式下的 ID

        elif self.mode in ['test', 'val']:
            im = []
            label = self.data[index, 1] - 1
            sub_id = self.data[index, 0]       # ✅ 验证/测试模式下的 ID
            for aug_num in range(self.cfg.DATA_AUG_NUM):
                nextimg = self.__load__(index, aug_num)
                step = 0
                while nextimg is None:
                    print(f'corrupted data {index} {aug_num}')
                    aug_num = random.randint(0, self.cfg.DATA_AUG_NUM - 1)
                    nextimg = self.__load__(index, aug_num)
                    step += 1
                    if step > self.num_retries:
                        break
                im.append(nextimg)

        # ✅ 在函数最后添加这个判断逻辑
        if getattr(self, "return_subid", False):
            return im, label, sub_id
        else:
            return im, label

    def __len__(self):
        return len(self.data)*self.cfg.DATA_AUG_NUM if self.mode == 'train' else len(self.data)

if __name__ == '__main__':
    from mvit.config.defaults import get_cfg
    cfg = get_cfg()
    cfg.merge_from_file('configs/MVITv2_mri.yaml')
    cfg.DATA.PATH_TO_DATA_DIR = '/data01/zixi/TractoFormer/TractoFormer-MVIT-main/new_500.csv'
    dataset = Tractoembedding(cfg,mode='train')
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=10,shuffle=True)
    iter_data = next(iter(dataloader))