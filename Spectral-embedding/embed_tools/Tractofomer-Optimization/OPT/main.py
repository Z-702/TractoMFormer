import sys
import argparse
import os
import mnist
import torch
from models.dgcnn import DGCNN
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import whitematteranalysis as wma
import fibers
from torch.utils.tensorboard import SummaryWriter
import time


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def convert_fiber_to_array(inputFile, numberOfFibers, fiberLength, numberOfFiberPoints):
    if not os.path.exists(inputFile):
        print("<wm_cluster_from_atlas.py> Error: Input file", inputFile, "does not exist.")
        exit()
    print("\n==========================")
    print("input file:", inputFile)

    if numberOfFibers is not None:
        print("fibers to analyze per subject: ", numberOfFibers)
    else:
        print("fibers to analyze per subject: ALL")

    number_of_fibers = numberOfFibers
    fiber_length = fiberLength
    print("minimum length of fibers to analyze (in mm): ", fiber_length)
    points_per_fiber = numberOfFiberPoints
    print("Number of points in each fiber to process: ", points_per_fiber)

    # read data
    print("<wm_cluster_with_DEC.py> Reading input file:", inputFile)
    pd = wma.io.read_polydata(inputFile)

    if fiber_length > 0:
        # preprocessing step: minimum length
        print(" Preprocessing by length:", fiber_length, "mm.")
        pd2 = wma.filter.preprocess(pd, fiber_length, return_indices=False, preserve_point_data=True,
                                    preserve_cell_data=True, verbose=False)
    else:
        pd2 = pd

    # downsampling fibers if needed
    if number_of_fibers is not None:
        print("Downsampling to ", number_of_fibers, "fibers.")
        input_data = wma.filter.downsample(pd2, number_of_fibers, return_indices=False, preserve_point_data=True,
                                           preserve_cell_data=True, verbose=False)
    else:
        input_data = pd2

    fiber_array = fibers.FiberArray()
    fiber_array.convert_from_polydata(input_data, points_per_fiber=numberOfFiberPoints)
    feat = np.dstack((abs(fiber_array.fiber_array_r), fiber_array.fiber_array_a, fiber_array.fiber_array_s))

    return input_data, feat

def calculate_predictions_embedding(model, dataloader, batch_size=1024, n_feature=2):
    print('calculate_predictions_embedding')

    model.eval()

    f_emb = torch.zeros(batch_size, n_feature)
    for idx, data in enumerate(dataloader):
        with torch.no_grad():
            fiber = data.to(device)
            emb = model(fiber)
            if idx == 0:
                f_emb = emb
            else:
                f_emb = torch.cat((f_emb, emb), dim=0)

    return f_emb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep learning method for generating embedding.')
    parser.add_argument(
        '-intract', action="store", dest="inputtract", default="",
        help='A file of whole-brain tractography as vtkPolyData (.vtk or .vtp).')
    parser.add_argument(
        '-l', action="store", dest="fiberLength", type=int, default=0,
        help='Minimum length (in mm) of fibers to analyze. ')
    parser.add_argument(
        '-p', action="store", dest="numberOfFiberPoints", type=int, default=14,
        help='Number of points in each fiber to process. 10 is default.')

    parser.add_argument('-mode', default="test", type=str, help='train or test')

    # training:
    parser.add_argument('-inputEmbed', default='./dataset/embed.npy',
        help='Input embedding npy.')

    # testing
    parser.add_argument(
        '-modelpath', action="store", dest="modelpath", default="./weights/Atlas-250.pt",
        help='Output folder of clustering results.')
    parser.add_argument(
        '-outdir',
        help='The output directory should be a new empty directory. It will be created if needed.')

    # other parameters
    parser.add_argument('--input_channel', default=3, type=int, help='input channel')
    parser.add_argument('--batch_size', default=2048, type=int, help='batch size')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--epochs', default=250, type=int, help='epochs number')
    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate for clustering')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=5, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--idx', default=True, type=str2bool, help='idx for dgcnn')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    modelpath = args.modelpath
    outputDir = args.outdir
    os.makedirs(outputDir, exist_ok=True)

    input_pd_fname = args.inputtract
    input_pd, x_array = convert_fiber_to_array(input_pd_fname,
                                               numberOfFibers=None,
                                               fiberLength=args.fiberLength,
                                               numberOfFiberPoints=args.numberOfFiberPoints)
    num_points = args.numberOfFiberPoints

    if args.idx:
        idxf=torch.zeros((num_points,args.k),dtype=torch.int64,device=device)
        if args.k==5:
            idxf[:,0]=torch.tensor(range(num_points))
            idxf[:, 1]=torch.tensor(range(num_points))-2
            idxf[:, 2] = torch.tensor(range(num_points)) - 1
            idxf[:, 3] = torch.tensor(range(num_points)) + 1
            idxf[:, 4] = torch.tensor(range(num_points)) + 2
        elif args.k==3:
            idxf[:,0]=torch.tensor(range(num_points))
            idxf[:, 1] = torch.tensor(range(num_points)) - 1
            idxf[:, 2] = torch.tensor(range(num_points)) + 1
        idxf[idxf<0]=0
        idxf[idxf>num_points-1]=num_points-1
        idx=idxf.repeat(args.batch_size,1,1)
        print(f'idx shape:\n{idx.shape}')
    else:
        idx=None

    if args.mode == 'train':
        print('Training...')
        embedding = np.load(args.inputEmbed)

        idxTotal = len(x_array)
        idxTrain = int(idxTotal * 0.8)
        print(f"Total fibers:\t{idxTotal}\n{idxTrain} for training...\t{idxTotal - idxTrain} for validating...")
        wbcTrain, wbcTest = x_array[:idxTrain, :], x_array[idxTrain:, :]
        embTrain, embTest = embedding[:idxTrain, :2], embedding[idxTrain:, :2]

        datasetTrain = mnist.Fiber_Emb(wbcTrain, embTrain, transform=transforms.Compose([transforms.ToTensor()]))
        dataloaderTrain = torch.utils.data.DataLoader(datasetTrain, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=4)
        datasetTest = mnist.Fiber_Emb(wbcTest, embTest, transform=transforms.Compose([transforms.ToTensor()]))
        dataloaderTest = torch.utils.data.DataLoader(datasetTest, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=4)

        model = DGCNN(args, idx=idx).double().to(device)
        print(f"Model:\t{str(model)}")

        model = nn.DataParallel(model)
        print(f"Let's use {torch.cuda.device_count()} GPUs!")

        if args.use_sgd:
            print("Using SGD...")
            opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
        else:
            print("Using Adam...")
            opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        schedular = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

        criterion = nn.MSELoss()

        best_loss = 9.9

        pt_path = "./weights/nAtlas-" + str(args.epochs) + ".pt"
        np_path = "./results/nAtlas-" + str(args.epochs) + ".npy"
        save_path = "./log/nAtlas-" + str(args.epochs)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        print(f'Checkpoint path: {pt_path}\nTensorboard path: {save_path}')
        writer = SummaryWriter(save_path)

        for epoch in range(args.epochs):

            ##########Train##########
            train_loss = 0.0
            count = 0.0
            model.train()
            train_bar = tqdm(dataloaderTrain, file=sys.stdout)

            for data, label in train_bar:
                data, label = data.to(device), label.to(device)

                opt.zero_grad()

                logits = model(data)
                # print(f'logits: {logits[:5,:]}\nlabels: {label[:5,:]}')
                loss = criterion(logits, label)
                loss.backward()
                opt.step()

                train_loss += loss.item() * args.batch_size
                count += args.batch_size

                train_bar.desc = "train epoch[{}/{}] loss:{:.5f}".format(epoch + 1,
                                                                         args.epochs,
                                                                         train_loss * 1.0 / count)
            writer.add_scalars(f'Epoch-{args.epochs}', {'Train Loss': train_loss * 1.0 / count}, epoch)
            ##########Train##########
            schedular.step()
            ##########Validate##########
            val_loss = 0.0
            count = 0.0
            model.eval()
            with torch.no_grad():
                val_bar = tqdm(dataloaderTest, file=sys.stdout)
                for data, label in val_bar:
                    data, label = data.to(device), label.to(device)

                    logits = model(data)
                    loss = criterion(logits, label)

                    val_loss += loss.item() * args.batch_size
                    count += args.batch_size

                    val_bar.desc = "valid epoch[{}/{}] loss:{:.5f}".format(epoch + 1,
                                                                           args.epochs,
                                                                           val_loss * 1.0 / count)
                if (val_loss * 1.0 / count) < best_loss:
                    best_loss = val_loss * 1.0 / count
                    print(f'Best loss: {best_loss}')
                    torch.save(model.state_dict(), pt_path)
                    
            writer.add_scalars(f'Epoch-{args.epochs}', {'Validate Loss': val_loss * 1.0 / count}, epoch)

        writer.close()
        print('Finished Training...')


        model = DGCNN(args).double()
        weights = torch.load(pt_path)
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        model.load_state_dict(weights_dict)
        model.eval()
        model = model.to(device)

        print('Start Calculating...')
        since = time.time()

        dataset = mnist.Fiber(x_array, transform=transforms.Compose([transforms.ToTensor()]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        emb = calculate_predictions_embedding(model, dataloader, args.batch_size)

        print(f"\nShape of embedding: {emb.shape}")
        # print(f"Type of the matrix: {type(emb)}")
        np_emb = emb.cpu().detach().numpy()
        print(f"Type of the numpy matrix: {type(np_emb)}")

        print(f'np_emb:\n{np_emb[:5, :]}')
        print(f'std:\n{embedding[:5, :2]}')

        elapsed = time.time() - since
        print('prediction time:', elapsed)
        
        input = torch.autograd.Variable(torch.from_numpy(np_emb))
        label = torch.autograd.Variable(torch.from_numpy(embedding[:, :2]))
        criterion = nn.MSELoss()
        print(f'loss: {criterion(input.float(), label.float())}')

        ##########Validate##########

    elif args.mode == 'test':
        # model = DGCNN(args, idx=idx).double()
        model = DGCNN(args).double()
        # print(f"Model:\t{str(model)}")

        weights = torch.load(modelpath)
        weights_dict = {}
        for k, v in weights.items():
            new_k = k.replace('module.', '') if 'module' in k else k
            weights_dict[new_k] = v
        model.load_state_dict(weights_dict)

        model.eval()
        model = model.to(device)

        since = time.time()

        dataset = mnist.Fiber(x_array, transform=transforms.Compose([transforms.ToTensor()]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        emb = calculate_predictions_embedding(model, dataloader, args.batch_size)

        print(f"\nShape of embedding: {emb.shape}")
        # print(f"Type of the matrix: {type(emb)}")
        np_emb = emb.cpu().detach().numpy()
        print(f"Type of the numpy matrix: {type(np_emb)}")

        print(f'np_emb:\n{np_emb[:5, :]}')
        # print(f'std:\n{embedding[:5, :2]}')

        elapsed = time.time() - since
        print('prediction time:', elapsed)

        input = torch.autograd.Variable(torch.from_numpy(np_emb))
        # label = torch.autograd.Variable(torch.from_numpy(embedding[:, :2]))
        # criterion = nn.MSELoss()
        # print(f'loss: {criterion(input.float(), label.float())}')

        # 做test的时候需要更改成对应样本的id
        print('Saving embedding...')
        np.save(os.path.join(outputDir, 'embed.npy'), np_emb)
        #np.save('/home/panyi/TractOPT/Tractofomer-data/sub-10321/data/tst-' + str(args.epochs) + '.npy', np_emb)
