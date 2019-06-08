'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import re

from tensorboardX import SummaryWriter

from models import *
from utils import progress_bar, init_logger

from tensorboardX import SummaryWriter

def findLastCheckpoint(save_dir, modelName):
    file_list = glob.glob(os.path.join(save_dir, modelName, modelName + '_epoch*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*_epoch(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

# Training
def train(net, epoch, args):
    ''' Training code 
    '''
    # Init the loggers - storage intermediate information 
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    acc = 100.*correct/total

    return net.state_dict, acc, train_loss

def test(net, epoch, args):
    '''
    Testing code
    '''
    #global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    return acc

if __name__ == "__main__":
    # Sparser 
    # Dictionary of supported networks 
    net_libs = {'VGG11':VGG, 'VGG13':VGG, 'VGG16':VGG, 'VGG19':VGG,  
                'ResNet18':ResNet18,'ResNet34':ResNet34, 'ResNet50':ResNet50, 'ResNet101':ResNet101,'ResNet152':ResNet152, 
                'PreActResNet18':PreActResNet18,'PreActResNet34':PreActResNet34, 'PreActResNet50':PreActResNet50, 
                    'PreActResNet18':PreActResNet101,'PreActResNet18':PreActResNet152, 
                'DenseNet121':DenseNet121, 'DenseNet161':DenseNet161, 'DenseNet169':DenseNet169, 'DenseNet201':DenseNet201, 
                'ResNeXt29_2x64d':ResNeXt29_2x64d, 'ResNeXt29_4x64d':ResNeXt29_4x64d, 'ResNeXt29_8x64d':ResNeXt29_8x64d, 
                    'ResNeXt29_16x64d':ResNeXt29_16x64d, 'ResNeXt29_32x4d':ResNeXt29_32x4d, 
                'DPN26':DPN26, 'DPN92':DPN92, 
                'ShuffleNetG2':ShuffleNetG2, 'ShuffleNetG3':ShuffleNetG3, 
                'SENet18': SENet18, 
                'ShuffleNetV2':ShuffleNetV2, 
                'GoogLeNet':GoogLeNet, 'MobileNet':MobileNet, 'MobileNetV2':MobileNetV2, 'EfficientNetB0':EfficientNetB0 }
    
    net_input = {}
    for i, item in enumerate(net_libs.items(), 1):
        if i < 4:
            net_input[item[0]] = item[0]
        elif item[0] == 'ShuffleNetV2':
            net_input[item[0]] = 1
        else:
            net_input[item[0]] = ''

    network_name = 'VGG11'

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', type=bool, default = True, help='resume from a previous checkpoint')
    parser.add_argument("--save_every", type=int, default=5,help="Number of training steps")
    parser.add_argument("--log_dir", type=str, default= "logs", help='path of log files')
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--data_dir", type=str, default='../data', help='Location to storage data')
    parser.add_argument("--save_every_epochs", type=int, default=1,	help="Number of training epochs to save state")
    parser.add_argument("--network_name", type=str, default=network_name,	help="Name of the network")
    parser.add_argument("--milestone", nargs=2, type=int, default=[25, 50], \
						help="When to decay learning rate; should be lower than 'epochs'")
    parser.add_argument("--epoch_max", "--e", type=int, default= 100, help="Number of total training epochs")
    args = parser.parse_args()
    save_dir = 'checkpoint/' + network_name 

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    args.log_dir = save_dir + '/' + args.log_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    writer = SummaryWriter(args.log_dir)
    logger = init_logger(args)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building ' + network_name + ' model ...')
    
    net = net_libs[network_name](net_input[network_name])    

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        
    best_acc = 0
    best_epoch = 0
    start_epoch = 0

    if args.resume:
        # Load checkpoint.     
        try:            
            checkpoint = torch.load(save_dir + '/' + network_name +  '_best.pth')    
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['best_acc']
            best_epoch = checkpoint['best_epoch']   
            start_epoch = checkpoint['epoch']

            print("Resume training " + network_name + " from epoch " + str(start_epoch))
        except:
            print("Start training " + network_name + " from epoch 0 ")
            
    # Perform traning and testing     

    for epoch in range(start_epoch, args.epoch_max):

        # Learning rate value scheduling according to args.milestone
        if epoch > args.milestone[1]:
        	current_lr = args.lr / 1000.
        elif epoch > args.milestone[0]:
            current_lr = args.lr / 10.
        else:
            current_lr = args.lr

		# set learning rate in optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        print('learning rate %f' % current_lr)


        net_state, acc_train, train_loss = train(net, epoch, args)
        acc_test = test(net, epoch, args)

        # Save each epoch  
        state = {
            'acc_test': acc_test,
            'acc_train': acc_train,
            'epoch': epoch,
            'best_epoch': best_epoch,
            'best_acc': best_acc
        }
    
        torch.save(state, save_dir + '/' + args.network_name +  '_epoch' + str(epoch) + '.pth')
        # Log the scalar values
        writer.add_scalar('acc_test', acc_test, epoch)
        writer.add_scalar('acc_train', acc_train, epoch)
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('current_lr', current_lr, epoch)
        writer.add_scalar('best_acc', best_acc, epoch)
        writer.add_scalar('best_epoch', best_epoch, epoch)

        # Save the best results 
        if acc_test > best_acc:
            #print('Saving best results ..')
            state['net'] = net.state_dict()                   
            torch.save(state, save_dir + '/' + args.network_name +  '_best.pth')
            best_acc = acc_test
            best_epoch = epoch 
