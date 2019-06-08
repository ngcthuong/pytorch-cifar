## Train CIFAR10 with PyTorch

## Additional Updates

### Training 
- Can be run using IDE such as visual code. 
- Can use terminar by 
```
	python main.py --networ_name VGG16
```

### New features 
1. Save file with all support network name 
2. Added save check point for every epoch
	- Best accuracy epoch is save at checkpoint/network/network_best.pth 
	- Option to save network at each epoch 
	- E.g. VGG11 training data is saved at checkpoint/VGG11/VGG11_epoch#no_epoch.pth
3. Learning rate is now controlled 	
4. Bring tensorboardx Support	
	- Using command tensorboard --logdir=log_dir --host localhost --port 8088
	- Open browser at http://localhost:8088/
	- Only save scalar values of training testing accuracy, loss, epoch, and learning rate 
	
4. Avaiabled trained data 
	- VGG11 
	
	
#### Todo

1. Pretrain data of all network 
	- 



## Original Readme 

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

#### Prerequisites
- Python 3.6+
- PyTorch 1.0+

#### Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |

#### Learning rate adjustment
I manually change the `lr` during training:
- `0.1` for epoch `[0,150)`
- `0.01` for epoch `[150,250)`
- `0.001` for epoch `[250,350)`

Resume the training with `python main.py --resume --lr=0.01`
