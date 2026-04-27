# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Train/Evaluation workflow."""

import pprint
import mvit.models.losses as losses
import mvit.models.optimizer as optim
import mvit.utils.checkpoint as cu
import mvit.utils.distributed as du
import mvit.utils.logging as logging
import mvit.utils.metrics as metrics
import mvit.utils.misc as misc
import numpy as np
import torch
from mvit.datasets import loader
from mvit.datasets.mixup import MixUp
from mvit.models import build_model
from mvit.utils.meters import EpochTimer, TrainMeter, ValMeter

logger = logging.get_logger(__name__)

class FusionModel(torch.nn.Module):
    def __init__(self,model_list):
        super(FusionModel,self).__init__()
        self.model_list = torch.nn.ModuleList(model_list)
    def forward(self, xs):
        pred_list = []
        for i, model in enumerate(self.model_list):
            pred_list.append(model(i))
        torch.mean(torch.stack(pred_list), 0)


def train_epoch(
    train_loader,
    model,
    optimizer,
    scaler,
    train_meter,
    cur_epoch,
    cfg,
):
    """
    Perform the training for one epoch.
    Args:
        train_loader (loader): training loader.
        model (model): the model to train.
        optimizer (optim): the optimizer to perform optimization on the model's
            parameters.
        scaler (GradScaler): the GradScaler to help perform the steps of gradient scaling.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            mvit/config/defaults.py
    """
    # Enable train mode.
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)

    if cfg.MIXUP.ENABLE:
        mixup_fn = MixUp(
            mixup_alpha=cfg.MIXUP.ALPHA,
            cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
            mix_prob=cfg.MIXUP.PROB,
            switch_prob=cfg.MIXUP.SWITCH_PROB,
            label_smoothing=cfg.MIXUP.LABEL_SMOOTH_VALUE,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )

    for cur_iter, (inputs, labels) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            for i in range(len(inputs)):
                for j in range(len(inputs[i])):
                    inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
            labels = labels.cuda()
        if cfg.MIXUP.ENABLE:
            inputs, labels = mixup_fn(inputs, labels)
        
        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            preds = model(inputs)
            # logger.info(f'1: Preds:{preds}, Label: {labels}')
            # Explicitly declare reduction to mean.
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

            # Compute the loss.
            if 'bce' in cfg.MODEL.LOSS_FUNC :
                loss = loss_fun(preds, labels[:,None].float())
            else:
                loss = loss_fun(preds, labels)

        # check Nan Loss.
        misc.check_nan_losses(loss)

        # Perform the backward pass.
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        # Update the parameters.
        scaler.step(optimizer)
        scaler.update()

        if cfg.MIXUP.ENABLE:
            _top_max_k_vals, top_max_k_inds = torch.topk(
                labels, 2, dim=1, largest=True, sorted=True
            )
        #     idx_top1 = torch.arange(labels.shape[0]), top_max_k_inds[:, 0]
        #     idx_top2 = torch.arange(labels.shape[0]), top_max_k_inds[:, 1]
        #     preds = preds.detach()
        #     preds[idx_top1] += preds[idx_top2]
        #     preds[idx_top2] = 0.0
            labels = top_max_k_inds[:, 0]
        if 'bce' in cfg.MODEL.LOSS_FUNC:
            preds = (preds > 0).float().squeeze(-1)
            num_topks_correct = [preds.eq(labels).float().sum(),]*2
        else:

            num_topks_correct = metrics.topks_correct(preds, labels, (1, 1))
            # logger.info(f'2: Preds:{preds}, Label: {labels}')
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])

        # Copy the stats from GPU to CPU (sync point).
        loss, top1_err, top5_err = (
            loss.item(),
            top1_err.item(),
            top5_err.item(),
        )
        
        mb_size = preds.size(0)                   #统计每一批次输入的数据
        num_errors = (top1_err / 100.0) * mb_size #利用top1_err*batch_size来计算错误的数据量
        logger.info(f"[Batch Eval] top1_err: {top1_err:.2f}%, batch_size: {mb_size}, top1_mis: {int(num_errors)}") #打印出来每一个batch的batch大小和错误的数据大小

        # Update and log stats.
        train_meter.update_stats(
            top1_err,
            top5_err,
            loss,
            lr,
            cfg.TRAIN.BATCH_SIZE,  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )

        train_meter.iter_toc()
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
        if cur_iter > 100:
            break

    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            mvit/config/defaults.py
    """
    logger.info(f"[Val Debug] Dataset size: {len(val_loader.dataset)}, Total iters: {len(val_loader)}")
    model.eval()
    val_meter.iter_tic()
    
    total_samples = 0
    total_errors = 0

    for cur_iter, (inputs, labels) in enumerate(val_loader):
        labels = labels.cuda(non_blocking=True)
        val_meter.data_toc()

        # ============================================================
        # 🧠 明确区分输入类型（防止误判多模态为多增强）
        # ============================================================
        if torch.is_tensor(inputs):
            mode = "single"
            inputs = inputs.cuda(non_blocking=True)

        elif isinstance(inputs, list) and all(torch.is_tensor(x) for x in inputs):
            mode = "fusion"
            for j in range(len(inputs)):
                inputs[j] = inputs[j].cuda(non_blocking=True)

        elif isinstance(inputs, list) and all(isinstance(x, list) for x in inputs):
            # ==============================
            # 🟧 多增强模式（可能含多模态）
            # ==============================
            mode = "augmentation"
            if isinstance(inputs[0][0], list):
                for i in range(len(inputs)):
                    for j in range(len(inputs[i])):
                        for k in range(len(inputs[i][j])):
                            inputs[i][j][k] = inputs[i][j][k].cuda(non_blocking=True)
            else:
                for i in range(len(inputs)):
                    for j in range(len(inputs[i])):
                        inputs[i][j] = inputs[i][j].cuda(non_blocking=True)

        else:
            raise ValueError(f"Unexpected input structure: {type(inputs)}")

        # ============================================================
        # 🚀 Forward pass (根据模式决定 forward 策略)
        # ============================================================
        if mode in ["single", "fusion"]:
            preds = model(inputs)

        elif mode == "augmentation":
            pred_list = []
            for aug in inputs:
                pred = model(aug)
                if 'bce' in cfg.MODEL.LOSS_FUNC:
                    index = (pred > 0.5).float().squeeze(-1)
                else:
                    _, index = torch.topk(pred, 1, dim=1, largest=True, sorted=True)
                pred_list.append(index.float())
            preds = torch.stack(pred_list, -1)
            preds = torch.mode(preds, -1)[0].reshape(-1)

        # ============================================================
        # 🎯 Compute error / accuracy / F1 (保持你的原逻辑)
        # ============================================================
        if 'bce' in cfg.MODEL.LOSS_FUNC:
            preds = (preds > 0.5).float().squeeze(-1)
        else:
            if preds.dim() == 2:
                preds = preds.argmax(preds,dim=1)
            if preds.dim() == 1:
                preds = preds.long()
            else:
                raise ValueError(f"Unexpected preds dimension: {preds.dim()}")
        num_topks_correct=[preds]
        num_topks_correct = [preds.eq(labels).float().sum(),] * 2

        mixed = [((labels==0)&(preds==labels)).float().sum(),
                 ((labels==0)&(preds!=labels)).float().sum(),
                 ((labels==1)&(preds==labels)).float().sum(),
                 ((labels==1)&(preds!=labels)).float().sum()]
        prec = mixed[0]/(mixed[0]+mixed[1]+1e-6)
        recall = mixed[0]/(mixed[0]+mixed[3]+1e-6)
        f1 = 2*prec*recall/(prec+recall+1e-6)
        logger.info(f'prec:{prec:.4f}, recall:{recall:.4f}, f1:{f1:.4f}, mode={mode}')

        correct = num_topks_correct[0]
        total = preds.size(0)
        logger.info(f"!!!![Epoch {cur_epoch}] Eval correct: {int(correct)}/{int(total)} | Accuracy: {(correct/total)*100:.2f}% (mode={mode})")

        top1_err, top5_err = [(1.0 - x / total) * 100.0 for x in num_topks_correct]
        if cfg.NUM_GPUS > 1:
            top1_err, top5_err = du.all_reduce([top1_err, top5_err])

        top1_err, top5_err = top1_err.item(), top5_err.item()
        batch_errors = (top1_err / 100.0) * total
        total_errors += batch_errors
        total_samples += total

        val_meter.iter_toc()
        val_meter.update_stats(top1_err, top5_err, cfg.TRAIN.BATCH_SIZE)
        val_meter.update_predictions(preds, labels)
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # ============================================================
    # 🔄 多 GPU 同步统计
    # ============================================================
    if cfg.NUM_GPUS > 1:
        total_errors_tensor = torch.tensor([total_errors], dtype=torch.float32, device=torch.cuda.current_device())
        total_samples_tensor = torch.tensor([total_samples], dtype=torch.float32, device=torch.cuda.current_device())
        torch.distributed.all_reduce(total_errors_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_samples_tensor, op=torch.distributed.ReduceOp.SUM)
        total_errors = total_errors_tensor.item()
        total_samples = total_samples_tensor.item()

    epoch_top1_err = (total_errors / total_samples) * 100.0
    val_meter.corrected_epoch_top1_err = epoch_top1_err
    if not hasattr(val_meter, "min_top1_err") or epoch_top1_err < val_meter.min_top1_err:
        val_meter.min_top1_err = epoch_top1_err

    logger.info(f"[Epoch {cur_epoch}] Final top1_err (based on error accumulation): {epoch_top1_err:.2f}%")
    correct_total = int(total_samples - total_errors)
    accuracy = (correct_total / total_samples) * 100
    logger.info(f"✅[Epoch {cur_epoch}] Full Eval correct: {correct_total}/{int(total_samples)} | Accuracy: {accuracy:.2f}%")

    val_meter.corrected_epoch_top1_err = 100 - accuracy
    val_meter.log_epoch_stats(cur_epoch)



def train(cfg):
    """
    Train a model on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in mvit/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    
    # Setup logging format.
    n_fold = cfg.N_FOLD
    logging.setup_logging(cfg.OUTPUT_DIR)
    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the model and print model statistics.
    model = build_model(cfg)

    
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)
    # Create a GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    # Load a checkpoint to resume training if applicable.
    start_epoch = cu.load_train_checkpoint(
        cfg, model, optimizer, scaler if cfg.TRAIN.MIXED_PRECISION else None
    )

    # Create the train and val loaders.
    train_loader = loader.construct_loader(cfg, "train",n_fold)
    val_loader = loader.construct_loader(cfg, "val",n_fold)
    logger.info(f"[Debug] 构造出的 val_loader.dataset 长度: {len(val_loader.dataset)}")
    
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    epoch_timer = EpochTimer()
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        loader.shuffle_dataset(train_loader, cur_epoch)

        # Train for one epoch.
        epoch_timer.epoch_tic()
        train_epoch(
            train_loader,
            model,
            optimizer,
            scaler,
            train_meter,
            cur_epoch,
            cfg,
        )
        epoch_timer.epoch_toc()
        logger.info(
            f"Epoch {cur_epoch} takes {epoch_timer.last_epoch_time():.2f}s. Epochs "
            f"from {start_epoch} to {cur_epoch} take "
            f"{epoch_timer.avg_epoch_time():.2f}s in average and "
            f"{epoch_timer.median_epoch_time():.2f}s in median."
        )
        logger.info(
            f"For epoch {cur_epoch}, each iteraction takes "
            f"{epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average. "
            f"From epoch {start_epoch} to {cur_epoch}, each iteraction takes "
            f"{epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
        )
        is_eval_epoch = misc.is_eval_epoch(cfg, cur_epoch)

        # Save a checkpoint.
        if is_checkp_epoch:
            cu.save_checkpoint(
                cfg.OUTPUT_DIR,
                model,
                optimizer,
                cur_epoch,
                cfg,
                scaler if cfg.TRAIN.MIXED_PRECISION else None,
            )
        # Evaluate the model on validation set.
        if is_eval_epoch:
            eval_epoch(val_loader, model, val_meter, cur_epoch, cfg)
    logger.info(f'The Final Min Err of fold {n_fold}:{val_meter.min_top1_err}',)

def test(cfg):
    """
    Perform testing on the pretrained model.
    Args:
        cfg (CfgNode): configs. Details can be found in mvit/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(pprint.pformat(cfg))

    # Build the model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    # Create meters.
    test_meter = ValMeter(len(test_loader), cfg)

    eval_epoch(test_loader, model, test_meter, -1, cfg)
    
if __name__ == '__main__':
    pass
    # preds_lst  = []
    # label = torch.randint(0,2,(20,1))
    # for _ in range(10):
    #     pred = torch.randn(20,2,)
    #     _, index = torch.topk(pred, 1, dim=1, largest=True, sorted=True)
    #     preds_lst.append(index.float())
    #     print(index)
    # preds = torch.stack(preds_lst, -1)
    # preds = (preds.mean(-1)>0.5).long()
    # print(preds.eq(label).float().sum())