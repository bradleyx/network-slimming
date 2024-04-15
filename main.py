from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models
import os
import tarfile
import pickle
from torch.utils.data import TensorDataset, DataLoader



# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=19, type=int,
                    help='depth of the neural network')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

# Load the data from pickle files
train_images = pickle.load(open('train_images.pkl', 'rb'))
train_labels = pickle.load(open('train_labels.pkl', 'rb'))
val_images = pickle.load(open('val_images.pkl', 'rb'))
val_labels = pickle.load(open('val_labels.pkl', 'rb'))

print("Type of val_images:", type(val_images))
print("Type of val_labels:", type(val_labels))

# Check the shape of val_images and val_labels
print("Shape of val_images:", val_images.shape)
print("Shape of val_labels:", val_labels.shape)

# Check the data type of val_images and val_labels
print("Data type of val_images:", val_images.dtype)
print("Data type of val_labels:", val_labels.dtype)

# Check the range of values in val_images
print("Minimum value in val_images:", val_images.min())
print("Maximum value in val_images:", val_images.max())

# Check the unique labels in val_labels
print("Unique labels in val_labels:", np.unique(val_labels))

# Define the transformations for your dataset
# train_transforms = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Pad(4),
#     transforms.RandomCrop(32),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
# ])

# test_transforms = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Pad(4),
#     transforms.RandomCrop(32),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
# ])


train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(4),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(4),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

# Apply transformations to the data
train_images_tensor = torch.stack([train_transforms(image) for image in train_images])
train_labels_tensor = torch.from_numpy(train_labels)
val_images_tensor = torch.stack([test_transforms(image) for image in val_images])
val_labels_tensor = torch.from_numpy(val_labels)

# Create TensorDataset instances
train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
val_dataset = TensorDataset(val_images_tensor, val_labels_tensor)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

# Get a batch of data from the train_loader
batch = next(iter(train_loader))
images, labels = batch

# Check the shapes of the tensors
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

# Check the data types of the tensors
print("Images dtype:", images.dtype)
print("Labels dtype:", labels.dtype)

# Check the range of values in the tensors
print("Images min value:", images.min())
print("Images max value:", images.max())
print("Labels min value:", labels.min())
print("Labels max value:", labels.max())

if args.refine:
    checkpoint = torch.load(args.refine)
    model = models.__dict__[args.arch]()  # Use your pruned model architecture
    
    # Load the state dictionary from the checkpoint
    state_dict = checkpoint['state_dict']
    
    # Create a new state dictionary with matching keys and shapes
    new_state_dict = {}
    for key, value in state_dict.items():
        if key in model.state_dict():
            if value.shape == model.state_dict()[key].shape:
                new_state_dict[key] = value
            else:
                print(f"Skipping parameter '{key}' due to shape mismatch: {value.shape} vs {model.state_dict()[key].shape}")
        else:
            print(f"Skipping parameter '{key}' as it is not present in the current model")
    
    # Load the new state dictionary into the model
    model.load_state_dict(new_state_dict, strict=False)
else:
    model = models.__dict__[args.arch]()

if args.cuda:
    model.cuda()
print(model.parameters())
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s*torch.sign(m.weight.data))  # L1

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        target=target.squeeze().long()
        loss = F.cross_entropy(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        loss.backward()
        if args.sr:
            updateBN()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        target=target.squeeze()
        print("Input data shape:", data.shape)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.h5'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.h5'), os.path.join(filepath, 'model_best.h5'))

        import h5py
        state_dict = model.state_dict()
        with h5py.File('model_weights.h5', 'w') as file:
            print("Writing to model_weights.h5")
            for layer_name, weights in state_dict.items():
                file.create_dataset(layer_name, data=weights.numpy())


best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):
    if epoch in [args.epochs*0.5, args.epochs*0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train(epoch)
    prec1 = test()
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best, filepath=args.save)

print("Best accuracy: "+str(best_prec1))