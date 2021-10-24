python train.py --dataset cifar100 --model resnet18 --data_augmentation --epochs 250

python train.py --dataset cifar100 --model resnet18 --data_augmentation  --lookahead --epochs 250 --pullback_momentum pullback

python train.py --SWA --epochs 250
python train.py --LA --epochs 250
python train.py  --lr_config SGD_config_lenet
# 

# python train.py --model lenet --epochs 50 --dataset mnist --K 1 --passport --lr_config SGD_config_lenet --sched_config Step_config
python train.py --model lenet --epochs 50 --dataset mnist --K 5 --passport --lr_config SGD_config_lenet --sched_config Step_config

python train.py --model resnet --epochs 200 --dataset cifar100 --K 1  --lr_config SGD_config --sched_config MultiStep_config
python train.py --model resnet --epochs 200 --dataset cifar100 --K 1 --passport --lr_config SGD_config --sched_config MultiStep_config

# python train.py --dataset cifar10 --model resnet18 --data_augmentation  --lookahead
# python train.py --dataset cifar10 --model resnet18 --data_augmentation
# python train.py --dataset cifar10 --model wideresnet --data_augmentation
# python train.py --dataset cifar10 --model wideresnet --data_augmentation --lookahead
# python train.py --dataset cifar100 --model wideresnet --data_augmentation
# python train.py --dataset cifar100 --model wideresnet --data_augmentation --lookahead



python train3.py --dir=swa --dataset=CIFAR100 --data_path=data --model=VGG16  --swa 