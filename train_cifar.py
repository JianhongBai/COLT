from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import transforms
from utils import setup_seed

from models.resnet import resnet10, resnet18, resnet50, wide_resnet50_2, resnet101, resnet152
from train_simCLR import cosine_annealing
import time
from utils import AverageMeter, logger
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import copy
from collections import OrderedDict

from data.cifar10 import subsetCIFAR10


parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('experiment', type=str, help='exp name')
parser.add_argument('--model', default='res18', type=str, help='model name')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--trainSplit', default='cifar10/trainIdxList.npy', type=str,
                    help='train split')
parser.add_argument('--valiSplit', default='', type=str,
                    help='validation split')
parser.add_argument('--fixto', default='nothing', type=str,
                    help='[nothing, layer1, layer2, layer3, layer4]')
parser.add_argument('--fixbn', action='store_true',
                    help='if specified, fix bn for the layers been fixed')
parser.add_argument('--checkpoint', default='', type=str,
                    help='path to pretrained model')
parser.add_argument('--resume', action='store_true',
                    help='if resume training')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='the start epoch number')
parser.add_argument('--decreasing_lr', default='75,90,95', help='decreasing strategy')
parser.add_argument('--cvt_state_dict', action='store_true', help='use for ss model')
parser.add_argument('--save-dir', default="checkpoints_tune", type=str)
parser.add_argument('--test_freq', default=1, help="test freq", type=int)
parser.add_argument('--test_only', action='store_true')
parser.add_argument("--cosineLr", action='store_true', help="if use cosine lr schedule")
parser.add_argument('--data', type=str, default='/path/to/cifar100', help='location of the data corpus')


args = parser.parse_args()
root = args.data
# settings
model_dir = os.path.join(args.save_dir, args.experiment)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
log = logger(os.path.join(model_dir))
log.info(str(args))
use_cuda = not args.no_cuda and torch.cuda.is_available()
setup_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
normalize_cifar100 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_cifar100
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_cifar100
])


if args.dataset == "cifar100":
    train_datasets = torchvision.datasets.CIFAR100(root=root, train=True, download=False, transform=transform_train)
    num_class = 100
    assert 'cifar100' in args.trainSplit
else:
    assert False

train_idx = list(np.load('split/{}'.format(args.trainSplit)))
train_sampler = SubsetRandomSampler(train_idx)
train_loader = torch.utils.data.DataLoader(
    train_datasets,
    batch_size=args.batch_size, sampler=train_sampler, num_workers=2)

if args.dataset == "cifar100":
    vali_datasets = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_test)
    if args.valiSplit == '':
        valiSplit = 'split/cifar100/cifar100_valIdxList.npy'
    elif args.valiSplit == 'test':
        valiSplit = None
        vali_datasets = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
    else:
        valiSplit = os.path.join('split', args.valiSplit)
    if valiSplit is not None:
        valid_idx = list(np.load(valiSplit))
    testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
else:
    assert False

if valiSplit is not None:
    valid_sampler = SubsetRandomSampler(valid_idx)
    vali_loader = torch.utils.data.DataLoader(
        vali_datasets,
        batch_size=args.batch_size, sampler=valid_sampler)
else:
    vali_loader = torch.utils.data.DataLoader(
        vali_datasets,
        batch_size=args.batch_size)

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def train(args, model, device, train_loader, optimizer, epoch, log, scheduler, label_map):
    model.train()
    if args.fixbn:
        fix_bn(model, args.fixto)

    dataTimeAve = AverageMeter()
    totalTimeAve = AverageMeter()
    end = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        dataTime = time.time() - end
        dataTimeAve.update(dataTime)

        optimizer.zero_grad()

        logits = model(data)
        loss = F.cross_entropy(logits, target)

        loss.backward()
        optimizer.step()
        if args.cosineLr:
            scheduler.step()
        totalTime = time.time() - end
        totalTimeAve.update(totalTime)
        end = time.time()
        # print progress
        if batch_idx % args.log_interval == 0:
            log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tData time: {:.3f}\tTotal time: {:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), dataTimeAve.avg, totalTimeAve.avg))


def eval_train(model, device, train_loader, log, label_map):
    model.eval()
    train_loss = 0
    correct = 0
    whole = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model.eval()(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            whole += len(target)
    train_loss /= len(train_loader.dataset)
    log.info('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, whole,
        100. * correct / whole))
    training_accuracy = correct / whole
    return train_loss, training_accuracy * 100


def eval_test(model, device, loader, log, epoch, num_class=10, prefix=''):
    model.eval()
    test_loss = 0
    correct = 0
    whole = 0
    perClassAccRight = [0 for _ in range(num_class)]
    perClassAccWhole = [0 for _ in range(num_class)]

    y_true = []
    y_pred = []

    with torch.no_grad():
        for data, target in loader:

            data, target = data.to(device), target.to(device)
            output = model.eval()(data)

            test_loss += F.cross_entropy(output, target, size_average=False).item()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            whole += len(target)

            y_true += target.view(-1).cpu().numpy().tolist()
            y_pred += pred.view(-1).cpu().numpy().tolist()

            for i in range(num_class):
                perClassAccRight[i] += pred[target == i].eq(target[target == i].view_as(pred[target == i])).sum().item()
                perClassAccWhole[i] += len(target[target == i])

    test_loss /= len(loader.dataset)

    log.info(prefix + 'Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, whole, 100. * correct / whole))

    msg = prefix + "Each class acc is "
    for i in range(num_class):
        msg += "{:.2f} ".format(perClassAccRight[i] / perClassAccWhole[i] * 100)

    accu_per_class = []
    for i in range(num_class):
        accu_per_class.append(perClassAccRight[i] / perClassAccWhole[i] * 100)

    log.info(msg)
    test_accuracy = correct / whole

    return test_loss, test_accuracy * 100


def fix_model(model, fixto):
    if fixto == 'nothing':
        # fix none
        # fix previous three layers
        for name, param in model.named_parameters():
            # print(name)
            param.requires_grad = True
    elif fixto == 'layer1':
        # fix previous three layers
        for name, param in model.named_parameters():
            # print(name)
            if not ("layer2" in name or "layer3" in name or "layer4" in name or "fc" in name):
                print("fix {}".format(name))
                param.requires_grad = False
            else:
                param.requires_grad = True
    elif fixto == 'layer2':
        # fix every layer except fc
        # fix previous four layers
        for name, param in model.named_parameters():
            # print(name)
            if not ("layer3" in name or "layer4" in name or "fc" in name):
                print("fix {}".format(name))
                param.requires_grad = False
            else:
                param.requires_grad = True
    elif fixto == 'layer3':
        # fix every layer except fc
        # fix previous four layers
        for name, param in model.named_parameters():
            # print(name)
            if not ("layer4" in name or "fc" in name):
                print("fix {}".format(name))
                param.requires_grad = False
            else:
                param.requires_grad = True
    elif fixto == 'layer4':
        # fix every layer except fc
        # fix previous four layers
        for name, param in model.named_parameters():
            # print(name)
            if not ("fc" in name):
                print("fix {}".format(name))
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        assert False


def fix_bn(model, fixto):
    if fixto == 'nothing':
        # fix none
        # fix previous three layers
        pass
    elif fixto == 'layer1':
        # fix the first layer
        for name, m in model.named_modules():
            if not ("layer2" in name or "layer3" in name or "layer4" in name or "fc" in name):
                m.eval()
    elif fixto == 'layer2':
        # fix the previous two layers
        for name, m in model.named_modules():
            if not ("layer3" in name or "layer4" in name or "fc" in name):
                m.eval()
    elif fixto == 'layer3':
        # fix every layer except fc
        # fix previous four layers
        for name, m in model.named_modules():
            if not ("layer4" in name or "fc" in name):
                m.eval()
    elif fixto == 'layer4':
        # fix every layer except fc
        # fix previous four layers
        for name, m in model.named_modules():
            if not ("fc" in name):
                m.eval()
    else:
        assert False


def main():
    # init model, ResNet18() can be also used here for training
    if args.model == 'res18':
        model = resnet18(pretrained=False, num_classes=num_class)

    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.cosineLr:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(step,
                                                    args.epochs * len(train_loader),
                                                    1,  # since lr_lambda computes multiplicative factor
                                                    1e-6 / args.lr,
                                                    warmup_steps=0)
        )
    else:
        decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)

    start_epoch = args.start_epoch

    if args.checkpoint != '' and not args.test_only:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'P_state' in checkpoint:
            state_dict = checkpoint['P_state']
        else:
            state_dict = checkpoint
        
        if args.cvt_state_dict:
            in_features = model.fc.in_features
            state_dict = cvt_state_dict(state_dict, args, in_features, num_classes=num_class)

        model.load_state_dict(state_dict)
        log.info('read checkpoint {}'.format(args.checkpoint))
        # eval_test(model, device, vali_loader, log)

        print("Testing validation accuracy")
        _, vali_tacc = eval_test(model, device, vali_loader, log, -1, num_class=num_class, prefix='Epoch: {} '.format(-1))

    elif args.resume:
        checkpoint = torch.load(os.path.join(model_dir, 'model.pt'))
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)


    if args.resume:
        if 'epoch' in checkpoint and 'optim' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            optimizer.load_state_dict(checkpoint['optim'])

            if args.cosineLr:
                for i in range(start_epoch * len(train_loader)):
                    scheduler.step()
            else:
                for i in range(start_epoch):
                    scheduler.step()
            log.info("resume the checkpoint {} from epoch {}".format(args.checkpoint, checkpoint['epoch']))
        else:
            log.info("cannot resume since lack of files")
            assert False

    ta = []
    best_prec1 = 0

    fix_model(model, args.fixto)

    if not args.test_only:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            # adjust learning rate for SGD
            if not args.cosineLr:
                scheduler.step()
            log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))

            # adversarial training
            train(args, model, device, train_loader, optimizer, epoch, log, scheduler, label_map=None)

            if epoch % args.test_freq == 0:
                # evaluation on natural examples
                print('================================================================')
                eval_train(model, device, train_loader, log, label_map=None)
                _, vali_tacc = eval_test(model, device, vali_loader, log, epoch, prefix='Epoch: {} '.format(epoch), num_class=num_class)
                ta.append(vali_tacc)
                print('================================================================')

                # save checkpoint
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                }, os.path.join(model_dir, 'model.pt'))

                is_best = vali_tacc > best_prec1
                best_prec1 = max(vali_tacc, best_prec1)

                if is_best:
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optim': optimizer.state_dict(),
                        'best_prec1': best_prec1,
                    }, os.path.join(model_dir, 'best_model.pt'))

    checkpoint = torch.load(os.path.join(model_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['state_dict'])
    _, test_tacc = eval_test(model, device, test_loader, log, 'best_model', num_class=num_class)
    log.info("On the best_model, test tacc is {}".format(test_tacc))


def cvt_state_dict(state_dict, args, in_features, num_classes):
    # deal with adv bn
    state_dict_new = copy.deepcopy(state_dict)

    name_to_del = []
    for name, item in state_dict_new.items():
        if 'bn_list' in name:
            name_to_del.append(name)
        if 'fc' in name:
            name_to_del.append(name)
    for name in np.unique(name_to_del):
        del state_dict_new[name]

    # deal with down sample layer
    keys = list(state_dict_new.keys())[:]
    name_to_del = []
    for name in keys:
        if 'downsample.conv' in name:
            state_dict_new[name.replace('downsample.conv', 'downsample.0')] = state_dict_new[name]
            name_to_del.append(name)
        if 'downsample.bn' in name:
            state_dict_new[name.replace('downsample.bn', 'downsample.1')] = state_dict_new[name]
            name_to_del.append(name)
    for name in np.unique(name_to_del):
        del state_dict_new[name]

    state_dict_new_noModule = OrderedDict()
    for name, item in state_dict_new.items():
        if 'module' in name:
            state_dict_new_noModule[name.replace('module.', '')] = item
    if len(state_dict_new_noModule.keys()) != 0:
        state_dict_new = state_dict_new_noModule


    state_dict_new['fc.weight'] = torch.zeros(num_classes, in_features).normal_(mean=0.0, std=0.01).to(state_dict_new['conv1.weight'].device)
    state_dict_new['fc.bias'] = torch.zeros(num_classes).to(state_dict_new['conv1.weight'].device)


    return state_dict_new


if __name__ == '__main__':
    main()
