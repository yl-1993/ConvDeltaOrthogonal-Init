import numpy as np
import torch
import os
import io
import pickle
from PIL import Image


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length):
        self.length = length
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0

    def update(self, val):
        self.history.append(val)
        if len(self.history) > self.length:
            del self.history[0]

        self.val = self.history[-1]
        self.avg = np.mean(self.history)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_ckpt(state, ckpt, epoch, is_best):
    folder = os.path.dirname(ckpt)
    fn = '{}_epoch_{}.pth.tar'.format(os.path.basename(ckpt), epoch)
    if folder != ''and not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, fn)
    print('saving to {}'.format(path))
    torch.save(state, '{}'.format(path))
    if is_best:
        best_fn = os.path.join(folder, 'model_best.pth.tar')
        if os.path.exists(best_fn):
            os.unlink(best_fn)
        os.symlink(fn, best_fn)
