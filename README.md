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
	
4. Avaiabled trained data, learning rate are at range [0 25 50 100 ] are 0.1 ,0.01, and 0.0001 (total 100 epochs). In the original repo, author use learning rate of 0.1, 0.01, and 0.001 for each 150 epochs (total 450 epochs). 
	
	
	
### Todo

1. Pretrain data of all network 


#### Accuracy 
At best test accuracy epoch

| Model             | Train Acc.  |  Test Acc.  |  Ref. Acc  | 
| ----------------- | :---: | :---: | :---: |
| [VGG11](https://arxiv.org/abs/1409.1556)  		   | 99.17%      | 90.92%      | 92.64%      |
| [VGG13](https://arxiv.org/abs/1409.1556)  		   | 99.49%      | 90.73%      | -      |
| [VGG16](https://arxiv.org/abs/1409.1556)  		   | 99.42%      | 90.76%      | -      |
| [VGG19](https://arxiv.org/abs/1409.1556)     		   | 99.15%      | 90.15%      | -      |
| [ResNet18](https://arxiv.org/abs/1512.03385) 		   | 99.76%      | 94.14%      | 93.02%      |
| [ResNet34](https://arxiv.org/abs/1512.03385) 		   | -%      | -%      | -      |
| [ResNet50](https://arxiv.org/abs/1512.03385) 		   | -%      | -%      | -      |
| [ResNet101](https://arxiv.org/abs/1512.03385) 	   | -%      | -%      | -      |
| [ResNet152](https://arxiv.org/abs/1512.03385) 	   | -%      | -%      | -      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)   | -%      | -%      | 95.11%      |
| [PreActResNet34](https://arxiv.org/abs/1603.05027)   | -%      | -%      | -      |
| [PreActResNet50](https://arxiv.org/abs/1603.05027)   | -%      | -%      | -      |
| [PreActResNet101](https://arxiv.org/abs/1603.05027)  | -%      | -%      | -      |
| [PreActResNet101](https://arxiv.org/abs/1603.05027)  | -%      | -%      | -      |
| [DenseNet121](https://arxiv.org/abs/1608.06993) 	   | -%      | -%      | 95.04%      |
| [DenseNet161](https://arxiv.org/abs/1608.06993)      | -%      | -%      | -      |
| [DenseNet169](https://arxiv.org/abs/1608.06993)      | -%      | -%      | -      |
| [DenseNet201](https://arxiv.org/abs/1608.06993)      | -%      | -%      | -      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1608.06993) | -%      | -%      | 94.82%      |
| [ResNeXt29(4x64d)](https://arxiv.org/abs/1608.06993) | -%      | -%      | -      |
| [ResNeXt29(8x64d)](https://arxiv.org/abs/1608.06993) | -%      | -%      | -      |
| [ResNeXt29(16x64d)](https://arxiv.org/abs/1608.06993) | -%     | -%      | -      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1608.06993) | -%      | -%      | 94.73%      |
| [DPN26](https://arxiv.org/abs/1707.01629)            | -%      | -%      | 95.16%      |
| [DPN92](https://arxiv.org/abs/1707.01629)            | -%      | -%      | -      |
| [MobileNet](https://arxiv.org/abs/1801.04381)        | -%      | -%      | -      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)      | -%      | -%      | 94.43%      |
| [ShuffleNetG2](https://arxiv.org/abs/1801.04381)     | -%      | -%      | -      |
| [ShuffleNetG3](https://arxiv.org/abs/1801.04381)     | -%      | -%      | -      |
| [ShuffleNetV2](https://arxiv.org/abs/1801.04381)     | -%      | -%      | -      |
| [GoogLeNet](https://arxiv.org/abs/1801.04381)        | -%      | -%      | -      |
| [EfficientNetB0](https://arxiv.org/abs/1801.04381)   | -%      | -%      | -      |
| [SENet18](https://arxiv.org/abs/1801.04381)     	   | -%      | -%      | -      |


In some case the results is not as good as expected like VGG19 perform less than VGG11. The reason might be learning rate and number of epoch is fixed for all networks. Also I trained only 100 epochs. The original repo is trained with 450 epoch. 


Ref. accuracy is reported in the original repo at [KuangLiu Repo](https://github.com/kuangliu/pytorch-cifar)


