### baseline

Epoch 000 - 099: bs=128 lr=0.1
Epoch 100 - 149: bs=128 lr=0.01
Epoch 149 - 199: bs=128 lr=0.001

python train_cifar10.py --num-epochs 200 --mode hybrid --gpu 2 -j 2 --batch-size 128 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --save-dir='baseline.2' --save-plot-dir='baseline.2'  2>&1 | tee cifar_resnet20_v1.log.0

best acc = 0.9183, 0.9185, 0.9181
mean = 0.9183

### 99_2x

Epoch 000 - 099: bs=128 lr=0.1
Epoch 100 - 149: bs=256 lr=0.02
Epoch 149 - 199: bs=256 lr=0.002

export NAME="99_2x.0"; python train_cifar10.py --num-epochs 100 --gpu 0,1 -j 2 --batch-size 128 --wd 0.0001 --lr 0.02 --lr-decay 0.1 --lr-decay-epoch 50 --model cifar_resnet20_v1 --save-dir=$NAME --save-plot-dir=$NAME --resume-from=baseline.0/cifar10-cifar_resnet20_v1-99.params 2>&1 | tee $NAME.log

### 99_4x

Epoch 000 - 099: bs=128 lr=0.1
Epoch 100 - 149: bs=512 lr=0.04
Epoch 149 - 199: bs=512 lr=0.004

export NAME="99_4x.0"; python train_cifar10.py --num-epochs 100 --gpu 0,1,2,3 -j 2 --batch-size 128 --wd 0.0001 --lr 0.04 --lr-decay 0.1 --lr-decay-epoch 50 --model cifar_resnet20_v1 --save-dir=$NAME --save-plot-dir=$NAME --resume-from=baseline.0/cifar10-cifar_resnet20_v1-99.params 2>&1 | tee $NAME.log
