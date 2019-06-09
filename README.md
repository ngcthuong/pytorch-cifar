## Train CIFAR10 with PyTorch

[PyTorch](http://pytorch.org/) on the CIFAR10 dataset with various up-to-date neural networks 

## Additional Updates

### Training 
- Can be run using IDE such as visual code. 
- Can use terminar by 
```
	python main.py --network_name VGG16
```
- Currently 

### New features 
1. Save file with all support network name 
2. Added save check point for every epoch
	- Best accuracy epoch is save at checkpoint/network/network_best.pth 
	- Option to save network at each epoch 
	- Network will automatic resum training 
	- Train from begining (i.e. epoch 0) by  
	```
		python main.py --network_name VGG16 --resume False 
	```
	- E.g. VGG11 training data is saved at checkpoint/VGG11/VGG11_epoch#no_epoch.pth
3. Learning rate is now controlled 	
4. Bring tensorboardx Support	
	- Using command  
	```
		tensorboard --logdir=log_dir --host localhost --port 8088
	```
	- Open browser at http://localhost:8088/
	- Only save scalar values of training testing accuracy, loss, epoch, and learning rate 
	
4. Avaiabled trained data, learning rate are at range [0 25 50 100 ] are 0.1 ,0.01, and 0.0001 
	- VGG11 
	
	
### Todo

1. Pretrain data of all network 


#### Accuracy 
At best test accuracy epoch

| Model             | Train Acc.  |  Test Acc.  |  Best Epoch  | 
| ----------------- | ----------- | ----------- | ------------ |
| [VGG11](https://arxiv.org/abs/1409.1556)  | 92.64%      | 92.64%      | 92.64%      |

Reported in the original repo at https://github.com/kuangliu/pytorch-cifar

| Model             | Train Acc.  |  
| ----------------- | ----------- |
| [VGG11](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |

