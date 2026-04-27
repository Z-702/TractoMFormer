import os, glob, shutil
from joblib import delayed, Parallel
import subprocess
import torch

#datafolder='/data01/BIDS_dataset/ds_140/'
#outdir='/data01/BIDS_dataset/tractoembedding'
datafolder = '/data01/zixi/vtk_PPMI'
outdir = '/data01/zixi/tractoembedding_PPMI_V2'
os.makedirs(outdir, exist_ok=True)

sub_ids = []
with open('/data01/zixi/filtered_PPMI_ids.txt', 'r') as f:
        for line in f:
                sub_ids.append(line.strip())
available_gpus = [0, 1]    #使用合适的空闲GPU           

def embed_with_gpu(idx, sub_id):
    gpu_id = available_gpus[idx % len(available_gpus)]  # 轮流分配 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[{sub_id}] Assigned to GPU {gpu_id}")
    embed_pipeline(sub_id, gpu_id)  # 调用你原本的函数

def embed_pipeline(sub_id, gpu_id):
        
        embedding_folder = os.path.join(outdir, sub_id)
        os.makedirs(embedding_folder, exist_ok=True)
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # 写入当前任务使用的 GPU 信息，方便你调试
        with open(os.path.join(embedding_folder, "gpu_debug.txt"), "w") as f:
                f.write(f"sub_id: {sub_id}\n")
                f.write(f"Assigned GPU: {gpu_id}\n")
                f.write(f"env['CUDA_VISIBLE_DEVICES']: {env['CUDA_VISIBLE_DEVICES']}\n")

        #tract_vtk = glob.glob(os.path.join(datafolder, f'{sub_id}*_run-1*vtp'))[0]
        #print(tract_vtk)
        tract_vtk = os.path.join(
            datafolder,
            "UKF_processed",
            f"{sub_id}_UKF",
            f"{sub_id}.vtk",
        )

        
        
        # DeepWMA
        embed_tract = os.path.join(embedding_folder, 'tracts', f'{sub_id}.vtp')
        embed_index = os.path.join(embedding_folder, 'tracts', f'{sub_id}.h5')
        
        while not os.path.exists(embed_tract):
                command = ['/data06/junyi/envs/DeepWMA/bin/python', '/data06/yijie/DeepWMA/dlt_extract_tract_feat.py', tract_vtk, os.path.join(embedding_folder, 'tracts'), '-outPrefix', sub_id, '-feature', 'RAS-3D', '-numPoints', '15']
                subprocess.check_call(command, env=env)

                command = ['/data06/junyi/envs/DeepWMA/bin/python', '/data06/yijie/DeepWMA/dlt_test.py', '/data06/yijie/DeepWMA/SegModels/CNN/cnn_model.h5', '-modelLabelName', '/data06/yijie/DeepWMA/SegModels/CNN/cnn_label_names.h5', os.path.join(embedding_folder, 'tracts', f'{sub_id}_featMatrix.h5'), os.path.join(embedding_folder, 'tracts'), '-outPrefix', sub_id, '-tractVTKfile', tract_vtk]
                subprocess.check_call(command, env=env)
                
                command = ['/data06/junyi/envs/tractoformer/bin/python', '/data06/yijie/TractoFormer/assess_cluster_location_by_hemisphere.py', os.path.join(embedding_folder, 'tracts', f'{sub_id}_prediction_tracts_outlier_removed'), '-clusterLocationFile', '/data06/yijie/TractoFormer/deepwma-hemi.csv', '-outputDirectory', os.path.join(embedding_folder, 'tracts', f'{sub_id}_separated')]
                subprocess.check_call(command, env=env)

                command = ['/data06/junyi/envs/tractoformer/bin/python', '/data06/yijie/TractoFormer/append_tracts.py', os.path.join(embedding_folder, 'tracts', f'{sub_id}_separated'), os.path.join(embedding_folder, 'tracts'), '-appendedTractName', sub_id]
                subprocess.check_call(command, env=env)

                os.remove(os.path.join(embedding_folder, 'tracts', f'{sub_id}_featMatrix.h5'))
                shutil.rmtree(os.path.join(embedding_folder, 'tracts', f'{sub_id}_prediction_tracts_outlier_removed'))
                shutil.rmtree(os.path.join(embedding_folder, 'tracts', f'{sub_id}_separated'))
                os.remove(os.path.join(embedding_folder, 'tracts', f'{sub_id}_multi_tract_specific_prediction_mask.h5'))
                
        embed_npy = os.path.join(embedding_folder, 'embed.npy')
        while not os.path.exists(embed_npy):
                command = ['/data06/junyi/envs/tractoformer/bin/python',  '/data06/yijie/Tractofomer-Optimization/OPT/main.py', '-intract', embed_tract, '-modelpath', '/data06/yijie/Tractofomer-Optimization/OPT/weights/Atlas-250.pt', '-mode', 'test', '-outdir', embedding_folder]
                subprocess.check_call(command, env=env)
                
        embed_diffusion_measure = os.path.join(embedding_folder, 'embed_feat.npy')
        while not os.path.exists(embed_diffusion_measure):
                command = ['/data06/junyi/envs/tractoformer/bin/python', '/data06/yijie/TractoFormer/extract_measures_tensors.py', '-inputVTK', embed_tract, '-outputNPY', embed_diffusion_measure]
                subprocess.check_call(command, env=env)
        
        # if not os.path.exists(os.path.join(embedding_folder, 'fiber_indices.npy')):
        while not os.path.exists(os.path.join(embedding_folder, 'tractoembedding/da-00010/')):
                try:
                        command = ['/data06/junyi/envs/tractoformer/bin/python', '/data06/yijie/TractoFormer/embedding_imaging.py', '-inputEmbed', embed_npy, '-inputMeasure', embed_diffusion_measure, '-outputdir', embedding_folder, '-prefix', sub_id, '-ds', '0.8', '-tractindexfile', embed_index]
                        subprocess.check_call(command, env=env)
                except Exception as e:
        print(f"[{sub_id}] embedding_imaging failed: {e}")
        raise
          
#tmp = Parallel(n_jobs=1, verbose=0, temp_folder='/data01/zixi/HCP_test')(delayed(embed_pipeline)(sub_id) for sub_id in sub_ids)

if __name__ == "__main__":
    Parallel(n_jobs=len(available_gpus), verbose=1)(
        delayed(embed_with_gpu)(i, sub_id) for i, sub_id in enumerate(sub_ids)
    )
