import logging
import time
import os
import copy
import operator

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

from DTI import models, dataset, cli, utils, analyse, visualizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def build_models_and_optimizers():
    nets = []
    for _ in range(len(args.INPUT_FEATURES)):
        if args.MODEL == '1D-CNN':
            nets.append(models.HARmodel(args).to(device))
        elif args.MODEL == '2D-CNN':
            nets.append(models.CNN_2D(args).to(device))
        elif args.MODEL == 'Lenet':
            nets.append(models.Lenet(args).to(device))
        elif args.MODEL == '1.5D':
            nets.append(models.new_CNN(args).to(device))
        else:
            raise ValueError(f"Unsupported MODEL: {args.MODEL}")

    optimizer = []
    for model in nets:
        optimizer.append(torch.optim.SGD(model.parameters(), lr=args.LR, weight_decay=args.L2))

    return nets, optimizer


def main():
    utils.setup_seed(123)
    start_time = time.time()

    vis = visualizer.Visualizer(args)

    LOG.info("Args:{}".format(args))

    fold_summaries = []

    # 直接使用 csv 里的 fold 列
    for fold_id in range(args.FOLD_NUM):
        LOG.info("========== Fold {}/{} ==========".format(fold_id + 1, args.FOLD_NUM))

        train_data = dataset.get_hcp_s1200(args, mode='train', n_fold=fold_id)
        val_data = dataset.get_hcp_s1200(args, mode='val', n_fold=fold_id)

        train_set = dataset.CreateDataset(args, train_data)
        val_set = dataset.CreateDataset(args, val_data)

        LOG.info(f"Fold {fold_id + 1}: train size = {len(train_set)}, val size = {len(val_set)}")

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            drop_last=False,
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers
        )

        nets, optimizer = build_models_and_optimizers()

        train_loss = []
        val_loss = []
        train_metric = []
        val_metric = []

        best_epoch = -1
        best_val_result = None
        best_val_metric = None

        for epoch in range(args.epochs):
            train_results = train(train_loader, nets, optimizer, epoch)
            val_results = validation(nets, val_loader)

            train_loss.append(train_results['loss'])
            val_loss.append(val_results['loss'])

            if args.target == 'sex':
                train_metric.append(train_results['acc'])
                val_metric.append(val_results['acc'])
                current_val_metric = val_results['acc']
                is_better = (best_val_metric is None) or (current_val_metric > best_val_metric)

                max_train_metric_index, max_train_metric = max(enumerate(train_metric), key=operator.itemgetter(1))
                max_val_metric_index, max_val_metric = max(enumerate(val_metric), key=operator.itemgetter(1))
                min_train_loss_index, min_train_loss = min(enumerate(train_loss), key=operator.itemgetter(1))

            elif args.target == 'age':
                train_metric.append(train_results['MAE'])
                val_metric.append(val_results['MAE'])
                current_val_metric = val_results['MAE']
                is_better = (best_val_metric is None) or (current_val_metric < best_val_metric)

                max_train_metric_index, max_train_metric = min(enumerate(train_metric), key=operator.itemgetter(1))
                max_val_metric_index, max_val_metric = min(enumerate(val_metric), key=operator.itemgetter(1))
                min_train_loss_index, min_train_loss = min(enumerate(train_loss), key=operator.itemgetter(1))

            else:
                raise ValueError("args.target 仅支持 'sex' 或 'age'")

            if is_better:
                best_val_metric = current_val_metric
                best_val_result = copy.deepcopy(val_results)
                best_epoch = epoch + 1

            print(
                'Fold {} | best train_loss:{}({}epoch) --- best t_metric:{}({}epoch) --- best val_metric:{}({}epoch)'.format(
                    fold_id + 1,
                    min_train_loss,
                    min_train_loss_index + 1,
                    max_train_metric,
                    max_train_metric_index + 1,
                    max_val_metric,
                    max_val_metric_index + 1
                )
            )

            vis.display_train_result(train_loss[-1], train_metric[-1], val_metric[-1], epoch)

        fold_summary = {
            'fold': fold_id + 1,
            'best_epoch': best_epoch,
            'best_val_result': best_val_result,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metric': train_metric,
            'val_metric': val_metric
        }
        fold_summaries.append(fold_summary)

    # 5-fold 汇总
    if args.target == 'sex':
        metric_name = 'acc'
    else:
        metric_name = 'MAE'

    best_metrics = [fold['best_val_result'][metric_name] for fold in fold_summaries]
    mean_metric = float(np.mean(best_metrics))
    std_metric = float(np.std(best_metrics))

    LOG.info("========== 5-Fold Summary ==========")
    for fold in fold_summaries:
        LOG.info(
            "Fold {} | best_epoch={} | best_val_{}={} | best_val_loss={}".format(
                fold['fold'],
                fold['best_epoch'],
                metric_name,
                fold['best_val_result'][metric_name],
                fold['best_val_result']['loss']
            )
        )

    LOG.info("5-Fold {} mean = {}".format(metric_name, mean_metric))
    LOG.info("5-Fold {} std  = {}".format(metric_name, std_metric))

    os.makedirs('./LOG', exist_ok=True)
    with open('./LOG/{}_5fold.txt'.format(args.RECORD_NAME), 'a+', encoding='utf-8') as f:
        f.writelines('args ' + str(args) + '\n')
        for fold in fold_summaries:
            f.writelines(
                'Fold {} | best_epoch {} | best_val_{} {} | best_val_loss {}\n'.format(
                    fold['fold'],
                    fold['best_epoch'],
                    metric_name,
                    fold['best_val_result'][metric_name],
                    fold['best_val_result']['loss']
                )
            )
        f.writelines('5fold_{}_mean {}\n'.format(metric_name, mean_metric))
        f.writelines('5fold_{}_std {}\n'.format(metric_name, std_metric))

    LOG.info("--- main.py finish in %s seconds ---" % (time.time() - start_time))


def train(dataloader, nets, optimizer, epoch):
    for model in nets:
        model.train()

    train_loss = 0
    pred_list = []
    target_list = []
    start_time = time.time()

    for batch_index, batch_samples in enumerate(dataloader):
        x, y = batch_samples['x'].to(device), batch_samples['y'].to(device)

        total_loss = 0
        output = np.zeros(1)

        for index, model in enumerate(nets):
            out = model(x[:, index:index + 1, :])

            if args.LOSS == 'CE':
                criteria = nn.CrossEntropyLoss()
                loss = criteria(out, y.long())
            elif args.LOSS == 'MSE':
                criteria = nn.MSELoss()
                loss = criteria(out.reshape(-1), y)
            else:
                raise ValueError("LOSS 仅支持 'CE' 或 'MSE'")

            optimizer[index].zero_grad()
            loss.backward()
            optimizer[index].step()

            total_loss += loss

            out = out.unsqueeze(2)
            if output.any():
                output = np.concatenate((output, out.detach().cpu().numpy()), axis=2)
            else:
                output = out.detach().cpu().numpy()

        if args.target == 'sex':
            output = np.argmax(output, axis=1)
            output = np.mean(output, axis=1)
            output = (output > 0.5).astype(int)
            pred = output
        elif args.target == 'age':
            output = np.mean(output, axis=2)
            output = np.squeeze(output, 1)
            pred = output

        loss_np = total_loss.detach().cpu().numpy()
        y_np = y.cpu().numpy()

        train_loss += loss_np
        pred_list = np.append(pred_list, pred)
        target_list = np.append(target_list, y_np)

        if batch_index % args.display_batch == 0:
            LOG.info("--- training progress rate {}/{} ---".format(batch_index, len(dataloader)))

    if args.target == 'sex':
        train_result = analyse.analyse_classification(train_loss, target_list, pred_list)
    elif args.target == 'age':
        train_result = analyse.analyse_regression(train_loss, target_list, pred_list)
    else:
        train_result = None

    LOG.info("--- training epoch {} finish in {} seconds ---".format(epoch, round(time.time() - start_time, 4)))
    return train_result


def validation(nets, val_loader):
    for model in nets:
        model.eval()

    test_loss = 0
    start_time = time.time()

    with torch.no_grad():
        pred_list = []
        target_list = []

        for batch_index, batch_samples in enumerate(val_loader):
            x, y = batch_samples['x'].to(device), batch_samples['y'].to(device)

            output = np.zeros(1)
            total_loss = 0

            for index, model in enumerate(nets):
                out = model(x[:, index:index + 1, :])

                if args.LOSS == 'CE':
                    criteria = nn.CrossEntropyLoss()
                    loss = criteria(out, y.long())
                elif args.LOSS == 'MSE':
                    criteria = nn.MSELoss()
                    loss = criteria(out.reshape(-1), y)
                else:
                    raise ValueError("LOSS 仅支持 'CE' 或 'MSE'")

                total_loss += loss

                out = out.unsqueeze(2)
                if output.any():
                    output = np.concatenate((output, out.detach().cpu().numpy()), axis=2)
                else:
                    output = out.detach().cpu().numpy()

            if args.target == 'sex':
                output = np.argmax(output, axis=1)
                output = np.mean(output, axis=1)
                output = (output > 0.5).astype(int)
                pred = output
            elif args.target == 'age':
                output = np.mean(output, axis=2)
                output = np.squeeze(output, 1)
                pred = output

            test_loss += total_loss.cpu().numpy()
            y_np = y.cpu().numpy()
            pred_list = np.append(pred_list, pred)
            target_list = np.append(target_list, y_np)

    if args.target == 'sex':
        val_result = analyse.analyse_classification(test_loss, target_list, pred_list)
    elif args.target == 'age':
        val_result = analyse.analyse_regression(test_loss, target_list, pred_list)
    else:
        val_result = None

    LOG.info("--- validation epoch finish in {} seconds ---".format(round(time.time() - start_time, 4)))
    return val_result


if __name__ == '__main__':
    LOG = logging.getLogger('main')
    logging.basicConfig(level=logging.INFO)
    args = cli.create_parser().parse_args()
    main()