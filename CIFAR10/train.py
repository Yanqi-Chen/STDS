import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from optim import Adam

import torchvision
from torchvision import transforms
from torch.cuda import amp

from spikingjelly.clock_driven.functional import reset_net
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven import surrogate

import numpy as np
import os
import sys
import time
import math
# from tqdm import tqdm

import model

############## Reproducibility ##############
_seed_ = 2020
np.random.seed(_seed_)
torch.manual_seed(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#############################################

args = None

def sineInc(n, N):
    return (1.0 + math.sin(math.pi * (float(n) / N - 0.5))) / 2
def linearInc(n, N):
    return float(n) / N

def main(args):

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    batch_size = args.batch_size
    learning_rate = args.lr
    dataset_dir = args.dataset_dir
    dump_dir = args.dump_dir
    T = args.T
    test = args.test
    i1 = args.interval_test
    N = args.epoch

    file_prefix = 'lr-' + np.format_float_scientific(learning_rate, exp_digits=1, trim='-') + f'-b-{batch_size}-T-{T}'

    file_prefix += f'-{args.sparse_function}'

    if ('flat' in args.sparse_function) or ('st' in args.sparse_function):
            file_prefix += f'-flat-{args.flat_width}'
            
    if args.gradual:
        file_prefix += f'-{args.gradual}'

    if args.amp:
        file_prefix += '-amp'

    log_dir = os.path.join(dump_dir, 'logs', file_prefix)
    model_dir = os.path.join(dump_dir, 'models', file_prefix)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=True,
        transform=transform_train,
        download=True)
    test_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=False,
        transform=transform_test,
        download=True)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True, 
        num_workers=4,
        pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False, 
        num_workers=4,
        pin_memory=True)

    total_train_step = len(train_data_loader) * N

    # Load existing model or create a new one
    net = model.__dict__['MultiStepCIFAR10Net'](multi_step_neuron=MultiStepLIFNode, tau=2.0, detach_reset=True, surrogate_function=surrogate.ATan(), backend='cupy').cuda()

    if args.alpha_gr == 0:
        optimizer = Adam(net.parameters(), lr=learning_rate)
    else:
        prune_params = []
        for name, module in net.named_modules():
            if isinstance(module, model.__dict__['PConv']) and module.weight.requires_grad:
                prune_params.append(module.weight)
        optimizer = Adam(
            [
                {
                    "params": prune_params,
                    "alpha_gr": args.alpha_gr,
                },
            ],
            lr=learning_rate
        )

    # Recover from unexpected breakpoint of training
    if os.path.exists(os.path.join(model_dir, 'checkpoint_latest.pth')):
        checkpoint = torch.load(os.path.join(model_dir, 'checkpoint_latest.pth'), map_location='cpu')
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'Load existing model, Train steps: {net.train_times}, Epochs: {net.epochs}')
        writer_test = SummaryWriter(log_dir + '/train', flush_secs=600, purge_step=net.epochs)
        writer_train = SummaryWriter(log_dir + '/test', flush_secs=600, purge_step=net.train_times)
    else:
        writer_test = SummaryWriter(log_dir + '/train', flush_secs=600)
        writer_train = SummaryWriter(log_dir + '/test', flush_secs=600)
        print(f'Create new model')     

    ###### TEST MODE ######
    if test:
        with torch.no_grad():
            test_sum = 0
            correct_sum = 0
            for img, label in test_data_loader:
                img = img.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                output = net(img, T).mean(0)

                correct_sum += (output.argmax(dim=1) == label).float().sum().item()
                test_sum += label.numel()

                reset_net(net)
            test_accuracy = correct_sum / test_sum
            print(f"Test Acc: {test_accuracy * 100:.2f}%")
            exit(0)

    ###### TRAIN MODE ######
    else:
        print(net)

        if args.amp:
            scaler = amp.GradScaler()
        else:
            scaler = None

        criterion = nn.MSELoss()
        
        # Training Loop
        for _ in range(N + 1):
            net.train()
            print(f'Epoch {net.epochs}, {file_prefix}')
            train_sum = 0
            correct_sum = 0
            train_loss = 0

            time_start = time.time()
            for img, label in train_data_loader:
                img = img.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

                optimizer.zero_grad()

                binary_label = F.one_hot(label, num_classes=10).float()

                if scaler is not None:
                    with amp.autocast():    
                        output = net(img, T).mean(0)
                        loss = criterion(output, binary_label)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = net(img, T).mean(0)
                    # print(output.shape, binary_label.shape)
                    # exit(0)
                    loss = criterion(output, binary_label)
                    loss.backward()
                    optimizer.step()                

                reset_net(net)

                correct_sum += (output.argmax(dim=1) == label).float().sum().item()
                train_sum += label.numel()
                train_loss += loss.item() * label.numel()

                net.train_times += 1

                # Threshold scheduler
                if args.gradual is not None:
                    for module in net.modules():
                        if hasattr(module, 'setFlatWidth'):
                            if args.gradual == 'linear':
                                module.setFlatWidth(linearInc(net.train_times, total_train_step) * args.flat_width)
                            elif args.gradual == 'sine':
                                module.setFlatWidth(sineInc(net.train_times, total_train_step) * args.flat_width)

            writer_train.add_scalar('train/acc', correct_sum / train_sum, net.train_times)
            writer_train.add_scalar('train/loss', train_loss / train_sum, net.train_times)
            
            # Evaluate at the end of training epoch
            net.eval()

            with torch.no_grad():

                test_sum = 0
                correct_sum = 0
                for img, label in test_data_loader:
                    img = img.cuda(non_blocking=True)
                    label = label.cuda(non_blocking=True)

                    output = net(img, T).mean(0)

                    correct_sum += (output.argmax(dim=1) == label).float().sum().item()
                    test_sum += label.numel()

                    reset_net(net)

                test_accuracy = correct_sum / test_sum
                print(f"Test Acc: {test_accuracy * 100:.2f}%")
                writer_test.add_scalar('test_acc', test_accuracy, net.epochs)

                torch.save(optimizer.state_dict(), os.path.join(model_dir, 'optim.pkl'))

                total_zerocnt = 0
                total_numel = 0
                for name, module in net.named_modules():
                    if hasattr(module, "getSparsity"):
                        zerocnt, numel = module.getSparsity()
                        total_zerocnt += zerocnt
                        total_numel += numel
                        print(f'{name}: {zerocnt / numel * 100:.2f}%')
                        writer_test.add_scalar(f'sparsity/{name}', zerocnt / numel, net.epochs)
                        if net.epochs % i1 == 0 or net.epochs == 0 or net.epochs == N:
                            writer_test.add_histogram(f'w/{name}', module.getSparseWeight(), net.epochs)
                            writer_test.add_histogram(f'theta/{name}', module.weight, net.epochs)
                print(f'total: {total_zerocnt / total_numel * 100:.2f}%')
                writer_test.add_scalar(f'sparsity/total', total_zerocnt / total_numel, net.epochs)

                checkpoint = {
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(model_dir, 'checkpoint_latest.pth'))

                if net.epochs % i1 == 0:
                    torch.save(net.state_dict(), os.path.join(model_dir, f'model-{net.epochs}.pth'))

            net.epochs += 1
            
            time_end = time.time()
            print(f'Elapse: {time_end - time_start:.2f}s')

            if net.epochs > N:
                break


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dataset-dir', type=str)
    parser.add_argument('--dump-dir', type=str)
    parser.add_argument('-T', type=int, default=8)
    parser.add_argument('-N', '--epoch', type=int, default=2048)
    parser.add_argument('-test', action='store_true')

    # Epoch interval when recording data (firing rate, acc. on test set, etc.) on TEST set
    parser.add_argument('-i1', '--interval-test', type=int, default=256)

    # Mixed precision training parameters
    parser.add_argument('--amp', action='store_true',
                        help='Use AMP training')

    # Added for Pruning
    parser.add_argument(
        "--sparse-function", type=str, choices=['identity', 'st', 'stmod'], default='identity', help="choice of reparameterization function")

    parser.add_argument(
        "--flat-width", type=float, default=1.0)
    parser.add_argument("--gradual", type=str, choices=['linear', 'sine'], default=None, help="increase type of flat width")

    parser.add_argument('--alpha-gr', default=0, type=float,
                        help='alpha in the Gradient Rewiring')

    args = parser.parse_args()

    return args

def run_args():
    global args
    if args is None:
        args = parse_args()
        
run_args()
if __name__ == "__main__":
    run_args()
    main(args)