import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import my_model  # Import your model

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming Prune')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = my_model()  # Initialize your model
if args.cuda:
    model.cuda()

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.model, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.model))

print(model)
total = 0
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        total += m.weight.data.numel()

conv_weights = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        size = m.weight.data.numel()
        conv_weights[index:(index+size)] = m.weight.data.view(-1).abs().clone()
        index += size

y, i = torch.sort(conv_weights)
thre_index = int(total * args.percent)
thre = y[thre_index]

pruned = 0
zero_flag = False
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.Conv2d):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.numel() - torch.sum(mask)
        m.weight.data.mul_(mask)
        if int(torch.sum(mask)) == 0:
            zero_flag = True
        print('layer index: {:d} \t total params: {:d} \t remaining params: {:d}'.
            format(k, mask.numel(), int(torch.sum(mask))))

print('Total conv params: {}, Pruned conv params: {}, Pruned ratio: {}'.format(total, pruned, pruned/total))

# Make real prune
# Delete all weight with mask 0
def actual_prune(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.weight.data = module.weight.data.masked_select(module.weight.data != 0).view(module.weight.data.size())
            if module.bias is not None:
                module.bias.data = module.bias.data.masked_select(module.bias.data != 0).view(module.bias.data.size())

if not zero_flag:
    actual_prune(model)

print(model)

torch.save({'state_dict': model.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))