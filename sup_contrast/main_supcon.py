from __future__ import print_function
import sys

import os
assert os.getcwd().split('/')[-1] == 'sup_contrast', "Sup contrast should be run from sup contrast directory and not from project root"
sys.path.append('.')

from utils import *
import os
import sys
import argparse
import time
import math

import wandb
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, #default=1000, we use moco hyperparameters
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.015, # default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120, 160', # default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--mean', type=str, default="(0.485, 0.456, 0.406)", help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, default="(0.229, 0.224, 0.225)", help='std of dataset in path in form of str tuple')

    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--fragment_file', type=str, default=None, help='path to fragment_file')
    parser.add_argument('--fragment_list_file', type=str, default=None, help="path to a text file containing list of fragment_file's")
    parser.add_argument('--n-samples', type=int, default=-1, help='n samples to use')
    parser.add_argument('--infinite_samples', action='store_true')
    parser.add_argument('--generation-resolution-multiplier', default=1.5, type=float, help='How extra big should the images be when generating them.')

    parser.add_argument('--size', type=int, default=128, help='parameter for RandomResizedCrop') # default 32
    parser.add_argument('--minimum-crop-area', type=float, default=0.02, help='parameter for RandomResizedCrop')   # default 0.2

    # save args
    parser.add_argument('--exp-name', type=str, default='test', help='experiment name')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR', 'CE'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    parser.add_argument('--random-seed', type=int, default=-1,
                        help='seed to use')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0')
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--gpus', type=str, default='3')

    opt = parser.parse_args()

    if opt.random_seed != -1:
        random.seed(opt.random_seed)
        np.random.seed(opt.random_seed)
        torch.manual_seed(opt.random_seed)

    select_gpus(opt.gpus)

    # check if dataset is path that passed required arguments
    assert int(opt.data_folder is not None) + int(opt.fragment_file is not None) + int(opt.fragment_list_file is not None), \
            "Only one of data_folder ({}), fragment_file ({}), or fragment_file_list ({}) must be provided.".format(opt.data_folder, opt.fragment_file, opt.fragment_list_file)
    assert opt.mean is not None and opt.std is not None

    # set the path according to the environment
    opt.model_path = './encoders/{}/{}'.format(opt.method, opt.exp_name)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)
    opt.model_name += '_' + str(opt.epochs)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
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


    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    print("Will save in " + opt.save_folder)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.log_path = os.path.join(opt.save_folder, 'log.txt')
    return opt

# parse before imports so that select_cuda_devices works
opt = parse_option()

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, SupCEResNet
from networks.smallalexnet import SupConSmallAlexNet, SupCESmallAlexNet
from networks.vit_base import SupConVitBase
from losses import SupConLoss

try:
    import apex
    from apex import amp, optimizers
    print("Successfully imported apex")
except ImportError:
    print("Failed to import apex")
    pass


def set_loader(opt):
    # construct data loader
    mean = eval(opt.mean)
    std = eval(opt.std)
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(opt.minimum_crop_area, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if not opt.data_folder is None:
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                                             transform=TwoCropTransform(train_transform))
        if opt.n_samples > 0 and 'single_fragments/' in opt.exp_name:
            assert len(train_dataset) >= opt.n_samples, "Requested {} samples but found {} in {}".format(opt.n_samples, len(train_dataset), opt.data_folder)
            final_samples = list(train_dataset.samples)
            final_samples.sort()
            train_dataset.samples = final_samples[:opt.n_samples]
            print("Training with {} samples as requested".format(len(train_dataset)))
        elif opt.n_samples > 0:
            assert opt.n_samples == len(train_dataset), "Subsampling only implemented for single_fragments experiment! And requested {} samples but dataset has {} samples!".format(opt.n_samples, len(train_dataset))

    else:
        from image_generation.shaders.on_the_fly_moderngl_shader import ModernGLOnlineDataset, get_sample_mixer

        if opt.infinite_samples:
            print("Training with infinite samples!")
            n_samples = -1
            virtual_dataset_size = opt.n_samples
        else:
            n_samples = opt.n_samples
            virtual_dataset_size = -1

        if not opt.fragment_file is None:
            fragment_files = [opt.fragment_file]
        else:
            fragment_files = read_text_file_lines(opt.fragment_list_file)

        # TODO: add parameter diversification for experiments
        parameter_diversity = False
        sample_mixer = None

        rendering_gpus = [int(k) for k in opt.gpus.split(',')]
        train_dataset = ModernGLOnlineDataset(fragment_files,
                                              TwoCropTransform(train_transform),
                                              sample_mixer=sample_mixer,
                                              parameter_diversity=parameter_diversity,
                                              resolution=int(opt.generation_resolution_multiplier * opt.size),
                                              n_samples=n_samples,
                                              virtual_dataset_size=virtual_dataset_size,
                                              gpus=rendering_gpus,
                                              base_shaders_path='../')

    print("Total training samples: {}".format(len(train_dataset)))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt, data_loader):
    if opt.method != 'CE':
        if 'vit-base' in opt.model:
            model = SupConVitBase()
        elif 'resnet' in opt.model:
            model = SupConResNet(name=opt.model)
        else:
            model = SupConSmallAlexNet(name=opt.model)
        criterion = SupConLoss(temperature=opt.temp)
    else:
        if 'vit-base' in opt.model:
            raise Exception("Needs to be implemented for CE with model vit-base!")
        elif 'resnet' in opt.model:
            model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
        else:
            model = SupCESmallAlexNet(name=opt.model, num_classes=opt.n_cls)
        criterion = torch.nn.CrossEntropyLoss()

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

global total_ii
total_ii = 0
def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    global total_ii

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        elif opt.method == 'CE':
            features = torch.cat((f1, f2), dim=0)
            labels = labels.repeat((2))
            loss = criterion(features, labels)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()
            wandb.log({'train/loss': float(loss.item()),
                       'lr/lr': optimizer.state_dict()['param_groups'][0]['lr'],
                       'iter_num': total_ii})
    total_ii += 1
    return losses.avg


def resume(opt, model, optimizer):

    epoch = 1
    if opt.resume:
        ckpt_path = opt.resume
    else:
        ckpt_path = os.path.join(opt.save_folder, 'checkpoint.pth')
        if not os.path.exists(ckpt_path):
            checkpoints_by_epoch = []
            for k in listdir(opt.save_folder, prepend_folder=True):
                try:
                    if 'ckpt_' in k and k.endswith('.pth'):
                        checkpoints_by_epoch.append((int(k.split('_')[-1].split('.')[0]), k))
                except Exception as e:
                    print("Failed to get last checkpoint by epoch with error:")
                    print(e)
                    pass
            checkpoints_by_epoch.sort()
            if len(checkpoints_by_epoch) > 0:
                ckpt_path = checkpoints_by_epoch[-1][-1]


    if os.path.exists(ckpt_path):
        print(f"Loading from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')

        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        epoch = ckpt['epoch']

    return epoch

def main(opt):
    save_file = os.path.join(opt.save_folder, 'last.pth')
    if os.path.exists(opt.save_folder + '/ckpt_epoch_{}.pth'.format(opt.epochs)):
        print("last checkpoint computed. Will exit!")
        exit(0)

    # build data loader
    print("Creating dataloader")
    train_loader = set_loader(opt)
    print("Finished creating dataloader")


    opt.n_cls = len(train_loader.dataset.class_to_idx.values())
    if opt.n_cls == 1 and not opt.method == 'SimCLR':
        raise Exception("Only one class detected in dataset, so method should be SimCLR (and method is {})".format(opt.method))
    print("Training with {} classes".format(opt.n_cls))
    print("Training with batch size {}".format(opt.batch_size))

    # build model and criterion
    model, criterion = set_model(opt, train_loader)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # resume
    start_epoch = resume(opt, model, optimizer)
    if start_epoch >= opt.epochs:
        print("Checkpoint already computed for epochs {}!".format(opt.epochs))
        exit()

    # wandb
    wandb.init(project='noise-learning-classification',
               name='supcon-{}-{}'.format(opt.model_name, opt.exp_name))

    for arg, argv in opt.__dict__.items():
        wandb.config.__setattr__(arg, argv)
        if 'SLURM_JOB_ID' in os.environ.keys():
            wandb.config.__setattr__('SLURM_JOB_ID', os.environ['SLURM_JOB_ID'])

    # watch model to log parameter histograms/gradients/...
    wandb.watch(model)

    # training routine
    for epoch in range(start_epoch, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)
        save_file = os.path.join(opt.save_folder, 'last.pth')
        if os.path.exists(save_file):
            print("Last checkpoint computed. Will exit!")
            exit(0)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        with open(opt.log_path, 'a') as f:
            print(f'Epoch: {epoch}, Loss: {loss}', file=f)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main(opt)
