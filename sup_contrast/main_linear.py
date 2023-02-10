from __future__ import print_function
import sys

import os
assert 'sup_contrast' in os.getcwd().split('/')[-1], "Sup contrast should be run from sup contrast directory and not from project root"

from utils import *

import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.resnet_big import SupConResNet, SupCEResNet, LinearClassifier
from networks.smallalexnet import SupConSmallAlexNet, SupCESmallAlexNet, SmallAlexnetLinearClassifier
from networks.vit_base import SupConVitBase, VitBaseLinearClassifier

from torchvision import transforms, datasets

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR', 'CE'], help='choose method')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--layer-index', type=int, default=5, help='layer index for small alexnet encoder')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--mean', type=str, default="(0.485, 0.456, 0.406)", help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, default="(0.229, 0.224, 0.225)", help='std of dataset in path in form of str tuple')

    parser.add_argument('--size', type=int, default=128, help='parameter for RandomResizedCrop')  # default 32
    parser.add_argument('--gpus', type=str, default='0')

    parser.add_argument('--n_samples', type=int, default=-1, help='samples to use from the dataset, will be sampled at random, if different than -1')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    select_gpus(opt.gpus)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    return opt


def set_loader(opt):
    # construct data loader
    mean = eval(opt.mean)
    std = eval(opt.std)

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(opt.size*1.1)),
        transforms.CenterCrop(opt.size),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = datasets.ImageFolder(root=opt.data_folder + '/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(root=opt.data_folder + '/val', transform=val_transform)
    
    if opt.n_samples > 0 and len(train_dataset) > opt.n_samples:
        r_state = random.getstate()
        random.seed(1337)
        train_dataset.samples = random.sample(train_dataset.samples, opt.n_samples)
        random.setstate(r_state)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader


def set_model(opt):
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if opt.model == 'small_alexnet':
        if opt.method != 'CE':
            model = SupConSmallAlexNet(name=opt.model)
        else:
            fc_key = 'module.fc.weight' if 'module.fc.weight' in state_dict.keys() else 'fc.weight'
            model = SupCESmallAlexNet(name=opt.model, num_classes=state_dict[fc_key].shape[0])
        classifier = SmallAlexnetLinearClassifier(num_classes=opt.n_cls, layer_index=opt.layer_index)
    elif 'vit-base' in opt.model:
        if opt.method != 'CE':
            model = SupConVitBase()
        else:
            raise Exception("Needs to be implemented for CE with model vit-base!")
        classifier = VitBaseLinearClassifier(num_classes=opt.n_cls)
    else:
        if opt.method != 'CE':
            model = SupConResNet(name=opt.model)
        else:
            fc_key = 'module.fc.weight' if 'module.fc.weight' in state_dict.keys() else 'fc.weight'
            model = SupCEResNet(name=opt.model, num_classes=state_dict[fc_key].shape[0])
        classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model.module.encoder)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            if opt.model == 'small_alexnet':
                features = model(images, layer_index=opt.layer_index)
            elif opt.model == 'vit-base':
                features = model.get_features(images)
            else:
                features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            if opt.model == 'small_alexnet':
                features = model(images, layer_index=opt.layer_index)
            elif opt.model == 'vit-base':
                features = model.get_features(images)
            else:
                features = model.encoder(images)
            output = classifier(features)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    best_accuracy_file = opt.ckpt + '_best_linear_accuracy' + opt.data_folder.replace('/', '_')
    print("Will save best accuracy file to :" + best_accuracy_file)
    if opt.n_samples > 0:
        best_accuracy_file += '_n_samples_' + str(opt.n_samples)
    if os.path.exists(best_accuracy_file):
        print("Linear evaluation already computed, will exit!")
        exit(0)

    opt.log_path = opt.ckpt + '_' + opt.data_folder.replace('/', '_') + '_n_samples_' + str(opt.n_samples) + '_linear_log.txt'
    # build data loader
    train_loader, val_loader = set_loader(opt)

    opt.n_cls = len(train_loader.dataset.class_to_idx.values())
    print("Training linear with {} classes".format(opt.n_cls))

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        if os.path.exists(best_accuracy_file):
            print("Accuracy file computed. Will exit!")
            exit(0)

        time2 = time.time()
        print_string = 'Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(epoch, time2 - time1, acc)
        print(print_string)
        with open(opt.log_path, 'a') as f:
            print(print_string, file=f)

        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc

    print('best accuracy: {:.2f}'.format(best_acc))
    write_text_file_lines(['best accuracy: {:.2f}'.format(best_acc)], best_accuracy_file)

if __name__ == '__main__':
    main()
