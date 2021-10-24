import os
from pprint import pprint

import torch
import torch.optim as optim
from torch import nn

# import passport_generator
from dataloader import prep_dataloader, toy_dataloader
from experiments.base import Experiment
from experiments.trainer import Trainer, Tester
# from experiments.utils import construct_passport_kwargs, load_passport_model_to_normal_model, \
#     load_normal_model_to_passport_model, load_normal_model_to_normal_model
from models.alexnet_normal import AlexNetNormal
from models.alexnet_passport import AlexNetPassport
from models.resnet_normal import ResNet18, ResNet18_pp, ResNet9
from models.lenet import LeNet, LeNet_passport, ToyNet
# from models.resnet import ResNet18
from models.resnet_passport import ResNet18Passport, ResNet9Passport
from optimizers.SWA import SWA
from optimizers.Lookahead import Lookahead

from configs import lr_configs


class ClassificationExperiment(Experiment):
    def __init__(self, args):
        super().__init__(args)

        self.train_datas, self.valid_data, self.size, self.in_channels, self.num_classes = prep_dataloader(self.args)
        # self.train_datas, self.valid_data = prepare_dataset(self.args)
        # self.train_datas, self.valid_data = toy_dataloader(self.args)

        self.construct_models()
        self.construct_optimizers()
        self.construct_lr_schedulers()
        self.trainer = Trainer(self.models, self.optimizers, self.schedulers, self.device)
        self.makedirs_or_load()


    def construct_lr_schedulers(self):
        self.schedulers = [getattr(optim.lr_scheduler, self.sched_config['scheduler'])(
            optimizer,
            **self.sched_config['sched_hparas']
        )for optimizer in self.optimizers]


    def construct_optimizers(self):
        # self.optimizers = [
        #     optim.SGD([
        #         {'params': model.encoder.parameters(), 'lr': 0.0001},
        #         {'params': model.net1.parameters()},
        #         {'params': model.net2.parameters()},
        #         ], **self.lr_config['optimizer']) 
        #     for model in self.models

        self.optimizers = [getattr(optim, self.lr_config['optimizer'])(
            model.parameters(), 
            **self.lr_config['optim_hparas']
        )for model in self.models]
        

    def construct_models(self):
        print('Construct Model ...')

        # def load_pretrained():
        #     if self.pretrained_path is not None:
        #         sd = torch.load(self.pretrained_path)
        #         model.load_state_dict(sd)

        if self.arch == 'alexnet':
            self.models = AlexNetNormal(self.in_channels, self.num_classes, self.norm_type)
        elif self.arch == 'lenet':
            if self.args['passport']:
                self.models = [LeNet_passport(self.in_channels, self.num_classes).cuda() for _ in range(self.K)]
            else:
                self.models = [LeNet(self.in_channels, self.num_classes).cuda() for _ in range(self.K)]
                # self.models = [ToyNet().cuda() for _ in range(self.K)]
        elif self.arch == 'resnet':
            if not self.args['passport']:
                ResNetClass = ResNet18 if self.arch == 'resnet' else ResNet9
                self.models = [ResNetClass(num_classes=self.num_classes, norm_type=self.norm_type).cuda() for _ in range(self.K)]
            else:
                self.models = [ResNet18_pp(num_classes=self.num_classes, norm_type=self.norm_type).cuda() for _ in range(self.K)]
        # load_pretrained()

        # pprint(self.model)


    def training(self):
        best_acc = float('-inf')

        history_file = os.path.join(self.logdir, 'history.csv')
        first = True

        if self.save_interval > 0:
            self.save_model('epoch-0.pth')

        print('Start training ...')

        for ep in range(1, self.epochs + 1):
            if self.args['K'] == 1:
                train_metrics = self.trainer.train_one(ep, self.train_datas[0])
            else:
                train_metrics = self.trainer.train(ep, self.train_datas)
            

            if ep % self.args['avg_freq'] == 0:
                self.trainer.Fed_avg()

            valid_metrics = self.trainer.test(self.valid_data, 'Testing Result')

            metrics = {'epoch': ep}
            # for key in train_metrics: metrics[f'train_{key}'] = train_metrics[key]
            for key in valid_metrics: metrics[f'valid_{key}'] = valid_metrics[key]

            self.append_history(history_file, metrics, first)
            first = False

            if self.save_interval and ep % self.save_interval == 0:
                self.save_model(f'epoch-{ep}.pth')

            if best_acc < metrics['valid_acc']:
                print(f'Found best at epoch {ep}\n')
                best_acc = metrics['valid_acc']
                # self.save_model('best.pth')

            # self.save_last_model()

    def evaluate(self):
        self.trainer.test(self.valid_data)


class AttackExperiment(Experiment):
    def __init__(self, args):
        super().__init__(args)

        self.train_datas, self.valid_data, self.size, self.in_channels, self.num_classes = prep_dataloader(self.args)

        self.construct_model()
        self.construct_optimizer()
        self.construct_lr_scheduler()
        self.trainer = Trainer(self.models, self.optimizers, self.schedulers, self.device)
        self.makedirs_or_load()


    def construct_lr_scheduler(self):
        self.schedulers = getattr(optim.lr_scheduler, self.sched_config['scheduler'])(
            self.optimizer,
            **self.sched_config['sched_hparas']
        )

    def construct_optimizer(self):
        self.optimizer = getattr(optim, self.lr_config['optimizer'])(
            self.model.parameters(), 
            **self.lr_config['optim_hparas']
        )
        

    def construct_model(self):
        print('Construct Model ...')

        # def load_pretrained():
        #     if self.pretrained_path is not None:
        #         sd = torch.load(self.pretrained_path)
        #         model.load_state_dict(sd)

        if self.arch == 'alexnet':
            self.models = AlexNetNormal(self.in_channels, self.num_classes, self.norm_type)
        elif self.arch == 'lenet':
            if self.args['passport']:
                self.model = LeNet_passport(self.in_channels, self.num_classes).cuda()
            else:
                self.model = LeNet(self.in_channels, self.num_classes).cuda()
                # self.models = [ToyNet().cuda() for _ in range(self.K)]
        elif self.arch == 'resnet':
            if not self.args['passport']:
                ResNetClass = ResNet18 if self.arch == 'resnet' else ResNet9
                self.model = ResNetClass(num_classes=self.num_classes, norm_type=self.norm_type).cuda()
            else:
                self.model = ResNet18_pp(num_classes=self.num_classes, norm_type=self.norm_type).cuda()
        # load_pretrained()

        # pprint(self.model)


    def training(self):
        best_acc = float('-inf')

        history_file = os.path.join(self.logdir, 'history.csv')
        first = True

        if self.save_interval > 0:
            self.save_model('epoch-0.pth')

        print('Start training ...')

        for ep in range(1, self.epochs + 1):
            if self.args['K'] == 1:
                train_metrics = self.trainer.train_one(ep, self.train_data)
            else:
                train_metrics = self.trainer.train(ep, self.train_datas)
            

            if ep % self.args['avg_freq'] == 0:
                self.trainer.Fed_avg()

            valid_metrics = self.trainer.test(self.valid_data, 'Testing Result')

            metrics = {'epoch': ep}
            # for key in train_metrics: metrics[f'train_{key}'] = train_metrics[key]
            for key in valid_metrics: metrics[f'valid_{key}'] = valid_metrics[key]

            self.append_history(history_file, metrics, first)
            first = False

            if self.save_interval and ep % self.save_interval == 0:
                self.save_model(f'epoch-{ep}.pth')

            if best_acc < metrics['valid_acc']:
                print(f'Found best at epoch {ep}\n')
                best_acc = metrics['valid_acc']
                # self.save_model('best.pth')

            # self.save_last_model()

    def evaluate(self):
        self.trainer.test(self.valid_data)
