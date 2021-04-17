# Another Back Propagation

   This is a repository illustrating the another back propagation about its utility, reproducibility and superiority.

## 1. Train model
   ```
   python train.py --model [str] --log [str] --lr [float] --momentum [float] --weight [float] --epoch [int] --train_batch_size [int] --eval_batch_size [int] [--use_wd]
   ```

   There are 3 types of comparable model, {BaselineModel, EmbeddingModel, MlpModel}, {Cnn, EmbeddingCnn, AbpCnn} and {Vgg16, Vgg16_bn, EmbeddingVgg16, EmbeddingVgg16_bn, AbpVgg16, AbpVgg16bn}. {*Model} and {*Cnn} is trained on MNIST, and {*VGG} on CIFAR-10.

1. train BaselineModel
    ```
    python train.py --model BaselineModel --log $number
    ```
2. train EmbeddingModel
    ```
    python train.py --model EmbeddingModel --log $number
    ```
3. train MlpModel
    ```
    python train.py --model MlpModel --log $number --weight 0.4
    ```
4. train Cnn
   ```
   python train.py --model Cnn --log $number --lr 0.05 --train_batch_size 64 --eval_batch_size 128
   ```
5. train  EmbeddingCnn
   ```
   python train.py --model EmbeddingCnn --log $number --lr 0.05 --train_batch_size 64 --eval_batch_size 128
   ```
6. train AbpCnn
   ```
   python train.py --model AbpCnn --log $number --lr 0.05 --weight 0.2 --train_batch_size 64 --eval_batch_size 128
   ```
7. train Vgg16
   ```
   python train.py --model Vgg16 --log $number --train_batch_size 128 --eval_batch_size 256 --lr 0.05 --epoch 100 --use_wd
   ```
8. train EmbeddingVgg16
   ```
   python train.py --model EmbeddingVgg16 --log $number --train_batch_size 128 --eval_batch_size 256 --lr 0.05 --epoch 100 --use_wd
   ```
9. train AbpVgg16
   ```
   python train.py --model AbpVgg16 --log $number --train_batch_size 128 --eval_batch_size 256 --weight 0.1 --lr 0.05 --epoch 100
   ```

Adjust epoch, learning rate, momentum, batch size as you like.

## 2. Requirements
pytorch >= 1.0

tensorboard

## 3. Results

training log organizes as the following form

```
├── log
│   ├── modelA  ├── number0  ├── run0.log
│   │                     
│   │           ├── number1  ├── run1.log
│   │          
│   │           ├── number2  ├── run2.log
│           
│   ├── modelB  ├── number0  ├── run0.log
│   │                      
│   │           ├── number1  ├── run1.log
│   │           
│   │           ├── number2  ├── run2.log
│         
│   ├── modelC  ├── number0  ├── run0.log
│   │                     
│   │           ├── number1  ├── run1.log
│   │           
│   │           ├── number2  ├── run2.log
│         
......
```

running
    ```
    tensorboard --logdir=log
    ```

you could get this:

   ![image](outcome0.png)

These 3 models above possesses the same network architecture, especially MlpModel amd BaselineModel is equivalent in parameter numbers. The smaller the training batch size is, the superiority of MlpModel is more apparent. However, when decreasing learning rate, their accuracy gap would decline, abp still performs not worse than others.

when set training batch size to 32, we would observe the following phenomenon:

   ![image](outcome1.png)

Image below doesn't show Cnn training curve, because it converge badly on the same settings, just try it yourself!

   ![image](outcome2.png)

ABP is obviously Robust！！！
