import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from models.resnet import resnet18, resnet10, resnet50, resnet101, resnet152, wide_resnet50_2
from utils import *
import torchvision.transforms as transforms
import torch.distributed as dist
from sklearn.cluster import KMeans as Kmeans_sklearn

import numpy as np
import copy
import math

from data.cifar100 import CustomCIFAR100
from optimizer.lars import LARS
from data.augmentation import GaussianBlur
from data.LT_Dataset import LT_Dataset
from data.LT_Dataset import Unsupervised_LT_Dataset

parser = argparse.ArgumentParser(description='')
parser.add_argument('experiment', type=str)
parser.add_argument('--save-dir', default='./checkpoints', type=str, help='path to save checkpoint')
parser.add_argument('--data', type=str, default='/path/to/cifar100', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar')
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', default=1000, type=int, help='number of total epochs to run')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--save_freq', default=100, type=int, help='save frequency /epoch')
parser.add_argument('--checkpoint', default='', type=str, help='saving pretrained model')
# parser.add_argument('--resume', default=False, type=bool, help='if resume training')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--optimizer', default='lars', type=str, help='optimizer type')
parser.add_argument('--lr', default=5.0, type=float, help='optimizer lr')
parser.add_argument('--scheduler', default='cosine', type=str, help='lr scheduler type')
parser.add_argument('--model', default='res18', type=str, help='model type')
parser.add_argument('--temperature', default=0.5, type=float, help='nt_xent temperature')
parser.add_argument('--output_ch', default=512, type=int, help='proj head output feature number')

parser.add_argument('--trainSplit', type=str, default='trainIdxList.npy', help="train split")
parser.add_argument('--imagenetCustomSplit', type=str, default='', help="imagenet custom split")

parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--local_rank', default=1, type=int, help='node rank for distributed training')

parser.add_argument('--strength', default=1.0, type=float, help='cifar augmentation, color jitter strength')
parser.add_argument('--resizeLower', default=0.1, type=float, help='resize smallest size')

parser.add_argument('--testContrastiveAcc', action='store_true', help="test contrastive acc")
parser.add_argument('--testContrastiveAccTest', action='store_true', help="test contrastive acc in test set")

parser.add_argument('--extra_type', type=str, default=None)
parser.add_argument('--COLT', action='store_true')
parser.add_argument('--warmup', type=int, default=100)
parser.add_argument('--sample_interval', type=int, default=25)
parser.add_argument('--budget', type=int, default=10000)
parser.add_argument('--sup_weight', type=float, default=0.2)
parser.add_argument('--beta', type=float, default=0.97)
parser.add_argument('--k_largest_logits', type=int, default=10)
parser.add_argument('--k_means_clusters', type=int, default=10)
parser.add_argument('--cluster_temperature', type=float, default=1.0)
parser.add_argument('--sample_set', type=str, default='')


def cosine_annealing(step, total_steps, lr_max, lr_min, warmup_steps=0):
    assert warmup_steps >= 0
    if step < warmup_steps:
        lr = lr_max * step / warmup_steps
    else:
        lr = lr_min + (lr_max - lr_min) * 0.5 * (
                    1 + np.cos((step - warmup_steps) / (total_steps - warmup_steps) * np.pi))
    return lr


def main():
    global args
    args = parser.parse_args()

    save_dir = os.path.join(args.save_dir, args.experiment)
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))

    print("distributing")
    dist.init_process_group(backend="nccl", init_method="env://")
    print("paired")
    torch.cuda.set_device(args.local_rank)

    rank = torch.distributed.get_rank()
    logName = "log.txt"

    log = logger(path=save_dir, local_rank=rank, log_name=logName)
    log.info(str(args))

    setup_seed(args.seed + rank)

    world_size = torch.distributed.get_world_size()
    print("employ {} gpus in total".format(world_size))
    print("rank is {}, world size is {}".format(rank, world_size))

    assert args.batch_size % world_size == 0
    batch_size = args.batch_size // world_size

    # define model
    if args.dataset == 'cifar100':
        imagenet = False
        num_class = 100
    elif args.dataset == 'imagenet-100':
        imagenet = True
        num_class = 100
    elif args.dataset == 'places365':
        imagenet = True
        num_class = 365

    if args.model == 'res18':
        model = resnet18(pretrained=False, imagenet=imagenet, num_classes=num_class)
    elif args.model == 'res50':
        model = resnet50(pretrained=False, imagenet=imagenet, num_classes=num_class)

    if model.fc is None:
        ch = 192
    else:
        ch = model.fc.in_features

    from models.utils import proj_head
    model.fc = proj_head(ch, args.output_ch)

    model.cuda()

    process_group = torch.distributed.new_group(list(range(world_size)))
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)

    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank, find_unused_parameters=True)

    cudnn.benchmark = True
    root = args.data

    if args.dataset == "cifar100":
        rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4 * args.strength, 0.4 * args.strength,
                                                                          0.4 * args.strength, 0.1 * args.strength)],
                                                  p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        normalize_cifar100 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        tfs_train = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(args.resizeLower, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            rnd_color_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize_cifar100
        ])

        tfs_test = transforms.Compose([
            transforms.ToTensor(),
            normalize_cifar100
        ])

        assert 'cifar100' in args.trainSplit
        train_idx = list(np.load('split/{}'.format(args.trainSplit)))
        train_datasets = CustomCIFAR100(train_idx, root=root, train=True, transform=tfs_train, download=False,
                                        returnIdx=True)
        from data.cifar100 import Supervised_CIFAR100
        # Note that we do NOT use the label information.
        train_datasets_test_trans = Supervised_CIFAR100(train_idx, root=root, train=True, transform=tfs_test,
                                                        download=False, return_targets=True, returnIdx=True)

        shuffle = True
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets, shuffle=shuffle)
        train_loader = torch.utils.data.DataLoader(
            train_datasets,
            num_workers=args.num_workers,
            batch_size=batch_size,
            sampler=train_sampler,
            pin_memory=True)

        train_loader_test_trans_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets_test_trans,
                                                                                          shuffle=False)
        train_loader_test_trans = torch.utils.data.DataLoader(
            train_datasets_test_trans,
            num_workers=args.num_workers,
            batch_size=batch_size,
            sampler=train_loader_test_trans_sampler,
            pin_memory=True,
        )

        if args.COLT:
            from data.random_images_loader import Unsupervised_RandomImages, Supervised_RandomImages
            sample_datasets_train_trans = Unsupervised_RandomImages(transform=transforms.Compose(
                [transforms.ToTensor(), transforms.ToPILImage(), tfs_train]), data_num=300000, return_ood_flag=True)
            sample_datasets_test_trans = Supervised_RandomImages(transform=transforms.Compose(
                [transforms.ToTensor(), transforms.ToPILImage(), tfs_test]), data_num=300000)
            ood_sampler = torch.utils.data.distributed.DistributedSampler(sample_datasets_test_trans, shuffle=False)
            sample_loader_test_trans = torch.utils.data.DataLoader(
                sample_datasets_test_trans,
                num_workers=args.num_workers,
                batch_size=batch_size,
                sampler=ood_sampler,
                shuffle=False,
                pin_memory=True,
                drop_last=True)

            print('Sample set total data number: {}'.format(len(sample_datasets_test_trans)))

    elif args.dataset == 'imagenet-100':

        rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        tfs_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            rnd_color_jitter,
            rnd_gray,
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            normalize_imagenet
        ])

        tfs_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize_imagenet
        ])

        txt = "split/imagenet-100/{}.txt".format(args.imagenetCustomSplit)
        print("use imagenet-100 {}".format(args.imagenetCustomSplit))

        train_datasets = Unsupervised_LT_Dataset(root=root, txt=txt, transform=tfs_train, returnIdx=True)
        train_datasets_test_trans = LT_Dataset(root=root, txt=txt, transform=tfs_test)

        shuffle = True
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets, shuffle=shuffle)
        train_loader = torch.utils.data.DataLoader(
            train_datasets,
            num_workers=args.num_workers,
            batch_size=batch_size,
            sampler=train_sampler,
            pin_memory=True)

        train_loader_test_trans_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets_test_trans,
                                                                                          shuffle=False)
        train_loader_test_trans = torch.utils.data.DataLoader(
            train_datasets_test_trans,
            num_workers=args.num_workers,
            batch_size=batch_size,
            sampler=train_loader_test_trans_sampler,
            pin_memory=True,
        )

        if args.COLT:
            if args.sample_set == 'imagenet-r':
                sample_txt = "split/imagenet-100/imagenet-r.txt"
                root = '/path/to/imagenet-r/'
                transform_test_imagenet = tfs_test
                imagenet_tfs_train = tfs_train
                sample_datasets_train_trans = Unsupervised_LT_Dataset(root=root, txt=sample_txt,
                                                                      transform=imagenet_tfs_train,
                                                                      return_ood_flag=True)
                sample_datasets_test_trans = LT_Dataset(root=root, txt=sample_txt, transform=transform_test_imagenet)
            ood_sampler = torch.utils.data.distributed.DistributedSampler(sample_datasets_test_trans, shuffle=False)
            sample_loader_test_trans = torch.utils.data.DataLoader(
                sample_datasets_test_trans,
                num_workers=args.num_workers,
                batch_size=batch_size,
                sampler=ood_sampler,
                shuffle=False,
                pin_memory=True,
                drop_last=True)

            print('Sample set total data number: {}'.format(len(sample_datasets_test_trans)))

    elif args.dataset == 'places365':
        from data.places365_loader import Unsupervised_places365_Dataset, Supervised_places365_Dataset
        rnd_color_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        tfs_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), interpolation=3),
            transforms.RandomHorizontalFlip(),
            rnd_color_jitter,
            rnd_gray,
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            normalize_imagenet
        ])

        tfs_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize_imagenet
        ])
        txt = "split/places365/Places_LT_train.txt"
        if args.imagenetCustomSplit != '':
            txt = "split/places365/{}.txt".format(args.imagenetCustomSplit)
        print("use places365 {}".format(args.imagenetCustomSplit))
        root = args.data
        train_datasets = Unsupervised_places365_Dataset(root=root, txt=txt, transform=tfs_train, returnIdx=True)
        train_datasets_test_trans = Supervised_places365_Dataset(root=root, txt=txt, transform=tfs_test)

        shuffle = True
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets, shuffle=shuffle)
        train_loader = torch.utils.data.DataLoader(
            train_datasets,
            num_workers=args.num_workers,
            batch_size=batch_size,
            sampler=train_sampler,
            pin_memory=True)

        train_loader_test_trans_sampler = torch.utils.data.distributed.DistributedSampler(train_datasets_test_trans,
                                                                                          shuffle=False)
        train_loader_test_trans = torch.utils.data.DataLoader(
            train_datasets_test_trans,
            num_workers=args.num_workers,
            batch_size=batch_size,
            sampler=train_loader_test_trans_sampler,
            pin_memory=True,
        )

        if args.COLT:
            if args.sample_set == 'places69':
                sample_txt = "split/places365/Places69_train.txt"
                root = "/path/to/places69/"
                transform_test_imagenet = tfs_test
                imagenet_tfs_train = tfs_train
                sample_datasets_train_trans = Unsupervised_places365_Dataset(root=root, txt=sample_txt,
                                                                      transform=imagenet_tfs_train,
                                                                      return_ood_flag=True)
                sample_datasets_test_trans = Supervised_places365_Dataset(root=root, txt=sample_txt, transform=transform_test_imagenet)
            ood_sampler = torch.utils.data.distributed.DistributedSampler(sample_datasets_test_trans, shuffle=False)
            sample_loader_test_trans = torch.utils.data.DataLoader(
                sample_datasets_test_trans,
                num_workers=args.num_workers,
                batch_size=batch_size,
                sampler=ood_sampler,
                shuffle=False,
                pin_memory=True,
                drop_last=True)

            print('Sample set total data number: {}'.format(len(sample_datasets_test_trans)))

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=args.lr, weight_decay=1e-6)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)
    else:
        print("no defined optimizer")
        assert False

    if args.scheduler == 'cosine':
        training_iters = args.epochs * len(train_loader)
        if args.COLT:
            training_iters = args.warmup * len(train_loader) + (args.epochs - args.warmup) * (
                        len(train_loader) + (args.budget // args.batch_size) + 1)
        warm_up_iters = 10 * len(train_loader)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(step,
                                                    training_iters,
                                                    1,  # since lr_lambda computes multiplicative factor
                                                    1e-6 / args.lr,
                                                    warmup_steps=warm_up_iters)
        )
    else:
        print("unknown schduler: {}".format(args.scheduler))
        assert False

    start_epoch = 1
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

    if args.resume:
        if args.checkpoint == '':
            checkpoint = torch.load(os.path.join(save_dir, 'model.pt'), map_location="cuda")
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            if 'epoch' in checkpoint and 'optim' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                optimizer.load_state_dict(checkpoint['optim'])

                for i in range((start_epoch - 1) * len(train_loader)):
                    scheduler.step()
                log.info("resume the checkpoint {} from epoch {}".format(args.checkpoint, checkpoint['epoch']))
            else:
                log.info("cannot resume since lack of files")
                assert False
        else:
            checkpoint = torch.load(args.checkpoint, map_location="cuda")
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            if 'epoch' in checkpoint and 'optim' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                optimizer.load_state_dict(checkpoint['optim'])

                for i in range((start_epoch - 1) * len(train_loader)):
                    scheduler.step()
                log.info("resume the checkpoint {} from epoch {}".format(args.checkpoint, checkpoint['epoch']))
            else:
                log.info("cannot resume since lack of files")
                assert False

    cls_ins_num = len(train_datasets)
    shadow = torch.zeros(len(train_datasets)).cuda()
    momentum_tail_score = torch.zeros(args.epochs, len(train_datasets)).cuda()
    for epoch in range(start_epoch, args.epochs + 1):
        log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        train_sampler.set_epoch(epoch)

        if args.COLT:
            if ((epoch-1) >= args.warmup and (epoch-1-args.warmup) % args.sample_interval == 0) or ((epoch - 1) == args.warmup):

                sample_idx = sample_ood(train_loader_test_trans, model, rank, world_size, sample_loader_test_trans,
                                        momentum_weight, args=args)
                ood_sample_subset = torch.utils.data.Subset(sample_datasets_train_trans, sample_idx.tolist())

                new_train_datasets = torch.utils.data.ConcatDataset([train_datasets, ood_sample_subset])
                train_sampler = torch.utils.data.distributed.DistributedSampler(new_train_datasets, shuffle=True)
                del train_loader
                train_loader = torch.utils.data.DataLoader(
                    new_train_datasets,
                    num_workers=args.num_workers,
                    batch_size=batch_size,
                    sampler=train_sampler,
                    pin_memory=True)

            shadow, momentum_tail_score = train(train_loader, model, optimizer, scheduler, epoch, log, args.local_rank,
                                          rank, world_size, shadow, momentum_tail_score,
                                          args=args)

            momentum_weight = momentum_tail_score[epoch - 1]

        else:
            shadow, momentum_tail_score = train(train_loader, model, optimizer, scheduler, epoch, log, args.local_rank,
                                          rank, world_size, shadow, momentum_tail_score,
                                          args=args)


        if rank == 0:
            save_model_freq = 2

            if epoch % save_model_freq == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim': optimizer.state_dict(),
                }, filename=os.path.join(save_dir, 'model.pt'))

            if epoch % args.save_freq == 0 and epoch > 0:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim': optimizer.state_dict(),
                }, filename=os.path.join(save_dir, 'model_{}.pt'.format(epoch)))

@torch.no_grad()
def sample_ood(train_loader_test_trans, model, rank, world_size, sample_loader_test_trans, momentum_weight=None, args=None):
    id_features = []
    id_idxs = []
    id_inputs = []

    model.eval()
    for i, (inputs, _, idx) in enumerate(train_loader_test_trans):
        d = inputs.size()
        inputs = inputs.cuda(non_blocking=True)
        idx = idx.view(-1).cuda(non_blocking=True)
        features = model.eval()(inputs)
        features_list = [torch.zeros_like(features) for _ in range(world_size)]
        torch.distributed.all_gather(features_list, features)
        features_list[rank] = features
        features = torch.cat(features_list)

        idx_list = [torch.zeros_like(idx) for _ in range(world_size)]
        torch.distributed.all_gather(idx_list, idx)
        idx_list[rank] = idx
        idxs = torch.cat(idx_list)

        if (i + 1) == len(train_loader_test_trans) and idxs.shape[0] < args.batch_size and world_size > 1:
            res = len(train_loader_test_trans.dataset) % world_size
            redundant_num = world_size - res
            res_tensor = torch.tensor([i for i in range(redundant_num)]).cuda()
            mask = torch.any(torch.eq(idxs.view(-1, 1), res_tensor.view(1, -1)), dim=1)
            assert mask.sum() == len(res_tensor)
            idxs = idxs[~mask]
            features = features[~mask]

        id_features.append(features.detach())
        id_idxs.append(idxs.detach())

        torch.distributed.barrier()
    torch.distributed.barrier()

    id_features = torch.cat(id_features)
    id_features = F.normalize(id_features, dim=-1)
    id_idxs = torch.cat(id_idxs)
    id_features = id_features[id_idxs.sort()[1]]
    momentum_weight = momentum_weight[:id_features.shape[0]]

    ood_features = []
    ood_idxs = []
    # torch.backends.cudnn.enabled = False
    for i, (inputs, _, idx) in enumerate(sample_loader_test_trans):
        d = inputs.size()
        inputs = inputs.cuda(non_blocking=True)
        idx = idx.view(-1).cuda(non_blocking=True)
        features = model.eval()(inputs)
        features_list = [torch.zeros_like(features) for _ in range(world_size)]
        torch.distributed.all_gather(features_list, features)
        features_list[rank] = features
        features = torch.cat(features_list)
        ood_features.append(features.detach())

        idx_list = [torch.zeros_like(idx) for _ in range(world_size)]
        torch.distributed.all_gather(idx_list, idx)
        idxs = torch.cat(idx_list)
        ood_idxs.append(idxs.detach())

        torch.distributed.barrier()
    torch.distributed.barrier()
    ood_features = torch.cat(ood_features)
    ood_features = F.normalize(ood_features, dim=-1)
    ood_idxs = torch.cat(ood_idxs)
    ood_features = ood_features[ood_idxs.sort()[1]]

    kmeans_features = id_features.detach().cpu().numpy()
    kmeans_ins = Kmeans_sklearn(n_clusters=args.k_means_clusters, random_state=0).fit(kmeans_features)
    cluster_labels = torch.from_numpy(kmeans_ins.labels_).cuda()
    cluster_weight = torch.zeros(args.k_means_clusters).cuda()

    for i in range(args.k_means_clusters):
        cluster_weight[i] = momentum_weight[cluster_labels == i].mean()

    cluster_weight = -((cluster_weight - torch.mean(cluster_weight)) / torch.std(cluster_weight))
    cluster_weight = F.softmax(cluster_weight / args.cluster_temperature, dim=0)
    cluster_budget = cluster_weight * args.budget
    cluster_budget = torch.tensor(cluster_budget, dtype=torch.int64)
    cluster_budget[-1] += int(args.budget - cluster_budget.sum())
    cluster_centers = torch.from_numpy(kmeans_ins.cluster_centers_)
    centers = F.normalize(cluster_centers.cuda(), dim=-1)
    distance_matrix, index_matrix = (1 - torch.mm(ood_features, centers.t())).t().sort(dim=1)
    sample_idx = []
    mask = torch.ones(distance_matrix.shape[1]).cuda()
    for i, cluster_i_budget in enumerate(cluster_budget):
        idx = index_matrix[i]
        idx = idx[mask[idx] != 0]
        cluster_i_sample_idx = idx[:cluster_i_budget]
        sample_idx.append(cluster_i_sample_idx)
        mask[cluster_i_sample_idx] = 0
    sample_idx = torch.cat(sample_idx)
    assert len(sample_idx) == len(torch.unique(sample_idx)) == args.budget
    return sample_idx

def train(train_loader, model, optimizer, scheduler, epoch, log, local_rank, rank, world_size, shadow=None,
          momentum_tail_score=None, args=None):
    losses = AverageMeter()
    losses.reset()
    data_time_meter = AverageMeter()
    train_time_meter = AverageMeter()
    end = time.time()
    id_count = 0
    for i, (inputs, index) in enumerate(train_loader):
        data_time = time.time() - end
        data_time_meter.update(data_time)

        index = index.cuda(non_blocking=True)
        idx_list = [torch.zeros_like(index) for _ in range(world_size)]
        torch.distributed.all_gather(idx_list, index)
        idx_list[rank] = index
        index = torch.cat(idx_list)

        d = inputs.size()
        inputs = inputs.view(d[0] * 2, d[2], d[3], d[4]).cuda(non_blocking=True)
        model.train()
        features = model(inputs)

        features_list = [torch.zeros_like(features) for _ in range(world_size)]
        torch.distributed.all_gather(features_list, features)
        features_list[rank] = features
        features = torch.cat(features_list)

        if (i + 1) == len(train_loader) and world_size > 1:
            res = world_size - (len(train_loader.dataset) % world_size)
            last_batch_each_gpu = math.ceil(len(index) / world_size)
            mask = torch.zeros_like(index, dtype=torch.bool)

            for j in range(world_size, res, -1):
                mask[last_batch_each_gpu * j - 1] = True

            index = index[~mask]
            features = features[(~mask).repeat(2)]

        neg_logits, loss_sample_wise, loss = nt_xent(features, t=args.temperature,
                                                                     index=index,
                                                                     sup_weight=args.sup_weight,
                                                                     args=args)
        neg_logits = neg_logits.mean(dim=0).detach()
        for count in range(features.shape[0] // 2):
            if not index[count] == -1:
                if epoch > 1:
                    new_average = (1.0 - args.beta) * neg_logits[count].sort(descending=True)[0][
                                                                    :args.k_largest_logits].sum().clone().detach() \
                                  + args.beta * shadow[index[count]]
                else:
                    new_average = neg_logits[count].sort(descending=True)[0][
                                  :args.k_largest_logits].sum().clone().detach()
                shadow[index[count]] = new_average
                momentum_tail_score[epoch - 1, index[count]] = new_average

        loss = loss * world_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(float(loss.detach().cpu() / world_size), inputs.shape[0])

        train_time = time.time() - end
        end = time.time()
        train_time_meter.update(train_time)

        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            log.info('Epoch: [{0}][{1}/{2}]\t'
                     'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                     'data_time: {data_time.val:.2f} ({data_time.avg:.2f})\t'
                     'train_time: {train_time.val:.2f} ({train_time.avg:.2f})\t'.format(
                epoch, i, len(train_loader), loss=losses,
                data_time=data_time_meter, train_time=train_time_meter))

    return shadow, momentum_tail_score


def save_checkpoint(state, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)


if __name__ == '__main__':
    main()


