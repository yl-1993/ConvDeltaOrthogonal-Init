import argparse
import torch.utils.data.distributed
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torchvision import datasets, transforms

import models
from utils import AverageMeter, accuracy, save_ckpt

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='M', help='mini-batch size (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-steps', default=[100, 150], type=list,
                    help='stpes to change lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='learing rate multiplier')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-path', default='checkpoints/ckpt', type=str,
                    help='path to store checkpoint (default: checkpoints)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on test set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--data-folder', type=str, default='./data',
                    help='where to store the datasets')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='dataset for training')
parser.add_argument('--conv-init', type=str, default='conv_delta_orthogonal',
                    help='Initializer for the convolution')
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    gpu_num = torch.cuda.device_count()
    print("=> using {} GPUS for training".format(gpu_num))
    if gpu_num > 0:
        args.cuda = True

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    path = os.path.join(args.data_folder, args.dataset)
    if args.dataset == 'mnist':
        num_classes = 10
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               normalize,
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               normalize,
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar10':
        """classes = ['plane', 'car', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        """
        num_classes = 10
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=path, train=True, download=True,
                             transform=transforms.Compose([
                             transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             normalize,
                         ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=path, train=False, download=True,
                             transform=transforms.Compose([
                             transforms.ToTensor(),
                             normalize,
                         ])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError('{} is not supported'.format(args.dataset))

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=num_classes, conv_init=args.conv_init, pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=num_classes, conv_init=args.conv_init)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    if args.evaluate:
        test(test_loader, model, criterion)
        return

    assert max(args.lr_steps) < args.epochs
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on test set
        prec1 = test(test_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_ckpt({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, args.save_path, epoch + 1, is_best)

    print('best test top-1 accuracy: {}'.format(best_prec1))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter(10)
    data_time = AverageMeter(10)
    losses = AverageMeter(10)
    top1 = AverageMeter(10)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, = accuracy(output, target, topk=(1,))
        losses.update(loss.item())
        top1.update(prec1[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'LR: {3}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader),
                   optimizer.param_groups[0]['lr'],
                   batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))


def test(test_loader, model, criterion):
    batch_time = AverageMeter(len(test_loader))
    losses = AverageMeter(len(test_loader))
    top1 = AverageMeter(len(test_loader))

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(test_loader):
            target = target.cuda(non_blocking=True)

            output = model(input)
            loss = criterion(output, target)

            prec1, = accuracy(output, target, topk=(1,))
            losses.update(loss.item())
            top1.update(prec1[0])

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       i, len(test_loader), batch_time=batch_time, loss=losses,
                       top1=top1))

        print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


if __name__ == '__main__':
    main()
