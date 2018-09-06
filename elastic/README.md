### baseline

Epoch 000 - 099: bs=128 lr=0.1
Epoch 100 - 149: bs=128 lr=0.01
Epoch 149 - 199: bs=128 lr=0.001

python train_cifar10.py --num-epochs 200 --mode hybrid --gpu 2 -j 2 --batch-size 128 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --save-dir='baseline.2' --save-plot-dir='baseline.2'  2>&1 | tee cifar_resnet20_v1.log.0

best acc = 0.9183, 0.9185, 0.9181
mean = 0.9183

```
The following results are invalid due to incorrect implementation of --resume-from option.

### 99_2x

Epoch 000 - 099: bs=128 lr=0.1
Epoch 100 - 149: bs=256 lr=0.02
Epoch 149 - 199: bs=256 lr=0.002

export NAME="99_2x.0"; python train_cifar10.py --num-epochs 100 --gpu 0,1 -j 2 --batch-size 128 --wd 0.0001 --lr 0.02 --lr-decay 0.1 --lr-decay-epoch 50 --model cifar_resnet20_v1 --save-dir=$NAME --save-plot-dir=$NAME --resume-from=baseline.0/cifar10-cifar_resnet20_v1-99.params 2>&1 | tee $NAME.log

best acc = 0.9197

### 99_4x

Epoch 000 - 099: bs=128 lr=0.1
Epoch 100 - 149: bs=512 lr=0.04
Epoch 149 - 199: bs=512 lr=0.004

export NAME="99_4x.0"; python train_cifar10.py --num-epochs 100 --gpu 0,1,2,3 -j 2 --batch-size 128 --wd 0.0001 --lr 0.04 --lr-decay 0.1 --lr-decay-epoch 50 --model cifar_resnet20_v1 --save-dir=$NAME --save-plot-dir=$NAME --resume-from=baseline.0/cifar10-cifar_resnet20_v1-99.params 2>&1 | tee $NAME.log

best acc = 0.9186
```

### mult-gpu baseline

#### 4 GPUs

M = 128
N = 4
lr = 0.4
best acc = 0.9190, 0.9184

python train_cifar10.py --num-epochs 200 --mode hybrid --gpus 0,1,2,3 -j 2 --batch-size 128 --wd 0.0001 --lr 0.4 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --save-dir='bs_512.0' --save-plot-dir='bs_512.0' 2>&1 | tee bs_512.0.log


#### 2 GPUs

M = 128
N = 2
lr = 0.2
best acc = 0.9206

python train_cifar10.py --num-epochs 200 --mode hybrid --gpus 0,1 -j 2 --batch-size 128 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --save-dir='bs_256.0' --save-plot-dir='bs_256.0' 2>&1 | tee bs_256.0.log;


### multi-gpu consistent batch size

#### 4 GPUs

M = 512
N = 1
lr = 0.4
best acc = 0.9155

```
batch_size=512, drop_rate=0.0, gpus='2', lr=0.4, lr_decay=0.1, lr_decay_epoch='100,150', lr_decay_period=0, mode='hybrid', model='cifar_resnet20_v1', momentum=0.9, num_epochs=200, num_workers=2, resume_from=None, save_dir='bs_512_gpu_1.0', save_period=10, save_plot_dir='bs_512_gpu_1.0', wd=0.0001
```

#### 2 GPUs

M = 256
N = 2
lr = 0.4
best acc = 0.9199

```
batch_size=256, drop_rate=0.0, gpus='0,1', lr=0.4, lr_decay=0.1, lr_decay_epoch='100,150', lr_decay_period=0, mode='hybrid', model='cifar_resnet20_v1', momentum=0.9, num_epochs=200, num_workers=2, resume_from=None, save_dir='bs_256_gpu_2_0.4.0', save_period=10, save_plot_dir='bs_256_gpu_2_0.4.0', wd=0.0001
```
