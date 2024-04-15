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
                    help='scale sparse rate (default : 0.5)')
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
        conv_weights[index:(index+size)] = m.weight.data.abs().clone().view(-1)
        index += size

y, i = torch.sort(conv_weights)
thre_index = int(total * args.percent)
thre = y[thre_index]

pruned = 0
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.Conv2d):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))

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

newmodel = my_model()
layer_id_in_cfg = 0
start_mask = torch.ones(3)
conv_idx = 0

for m0 in model.modules():
    if isinstance(m0, nn.Conv2d):
        end_mask = conv_weights[conv_idx:conv_idx+m0.weight.data.shape[0]].gt(thre).float().cuda()
        conv_idx += m0.weight.data.shape[0]
        break
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.Conv2d):
        print(f"Processing convolutional layer: {m0}")
        print(f"Weight shape: {m0.weight.data.shape}")
        print(f"start_mask: {start_mask}")
        print(f"end_mask: {end_mask}")
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        print(f"idx0: {idx0}")
        print(f"idx1: {idx1}")
        assert idx0.size <= m0.weight.data.shape[1], "idx0 size exceeds input channels"
        assert idx1.size <= m0.weight.data.shape[0], "idx1 size exceeds output channels"
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        w1 = w1[idx1.tolist(), :, :, :].clone()
        m1.weight.data = w1.clone()
        start_mask = end_mask.clone()
        layer_id_in_cfg = np.argwhere(np.asarray(list(model.modules())) == m0)[0][0]
        if layer_id_in_cfg < len(list(model.modules())) - 1 and isinstance(list(model.modules())[layer_id_in_cfg + 1], nn.Conv2d):
            end_mask = conv_weights[layer_id_in_cfg + 1].gt(thre).float().view(-1).cuda()
    elif isinstance(m0, nn.Linear):
        m1.weight.data = m0.weight.data.clone()
        m1.bias.data = m0.bias.data.clone()

print(model)
torch.save({'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))