import argparse
from pprint import pprint

import torch


from experiments.classification import ClassificationExperiment



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='resnet', choices=['lenet', 'alexnet', 'resnet', 'resnet9'],
                        help='architecture (default: resnet)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='training epochs (default: 200)')

    parser.add_argument('--dataset', default='cifar100', choices=['mnist',
                                                                'cifar10',
                                                                 'cifar100',
                                                                 'caltech-101',
                                                                 'caltech-256',
                                                                 'imagenet1000'],
                        help='training dataset (default: cifar10)')
    parser.add_argument('--norm-type', default='bn', choices=['bn', 'gn', 'in', 'none'],
                        help='norm type (default: bn)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')

    parser.add_argument('--passport', action='store_true', default=False,
                        help='enables passport')

    # optimizer
    parser.add_argument('--lr_config', default='SGD_config',
                        help='lr config python file')
    parser.add_argument('--sched_config', default='MultiStep_config',
                        help='lr_scheduler config python file')
            
    # misc
    parser.add_argument('--save-interval', type=int, default=0,
                        help='save model interval')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='for evaluation')
    parser.add_argument('--exp-id', type=int, default=1,
                        help='experiment id')
    parser.add_argument('--tag',
                        help='tag')


    args = parser.parse_args()

    pprint(vars(args))

    # exp = ClassificationExperiment(vars(args))
    exp = AttackExperiment(vars(args))


    exp.training()

    print('Training done at', exp.logdir)


if __name__ == '__main__':
    main()
