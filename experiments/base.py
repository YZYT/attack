import csv
import json
import os
from configs import lr_configs
import torch
from experiments.utils import init_Seed

class Experiment(object):
    """
    1. load variables
    2. load dataset
    3. load model
    4. load optimizer
    5. load trainer
    6. self.makedirs_or_load(args)
    """

    def __init__(self, args):
        init_Seed(args['seed'])

        self.is_tl = False

        self.args = args
        self.prefix = ''
        self.models = None
        self.trainer = None
        self.train_loader = None
        self.val_loader = None
        self.experiment_id = args['exp_id']

        self.buffer = []
        self.save_history_interval = 1
        self.device = torch.device('cuda')

        self.K = args['K']

        self.arch = args['model']
        self.dataset = args['dataset']
        self.epochs = args['epochs']
        self.batch_size = args['batch_size']

        self.eval = args['eval']
        self.tag = args['tag']
        self.save_interval = args['save_interval']
        self.lr_config = getattr(lr_configs, args['lr_config'])
        self.sched_config = getattr(lr_configs, args['sched_config'])
        self.pretrained_path = args['pretrained_path']

        self.norm_type = args['norm_type']

        self.logdir = f'logs/{self.arch}_{self.dataset}_{self.arch}'

        if self.tag is not None:
            self.logdir += f'_{self.tag}'

        self.imgcrop = 224 if self.dataset == 'imagenet1000' else 32

    def get_expid(self, logdir, prefix):
        exps = [d.replace(prefix, '') for d in os.listdir(logdir) if
                os.path.isdir(os.path.join(logdir, d)) and prefix in d]
        files = set(map(int, exps))
        if len(files):
            return min(set(range(1, max(files) + 2)) - files)
        else:
            return 1

    def finetune_load(self):
        # create directory like this: logdir/tl_{expid}

        self.prefix = 'tl_'
        self.logdir = os.path.join(self.logdir, str(self.experiment_id))

        path = os.path.join(self.logdir, 'models', 'best.pth')
        if not os.path.exists(path):
            print(f'Warning: No such Experiment -> {path}')
        else:
            print(f'Loading from {path}')
            self.load_model('best.pth')

        self.finetune_id = self.get_expid(self.logdir, self.prefix)

        self.logdir = os.path.join(self.logdir, f'{self.prefix}{self.finetune_id}')

        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(os.path.join(self.logdir, 'models'), exist_ok=True)

        print(f'Finetune logdir: {self.logdir}')

        json.dump(self.args, open(os.path.join(self.logdir, 'config.json'), 'w'), indent=4)
        self.model = self.model.to(self.device)

    def makedirs_or_load(self):
        # create directory like this: logdir/{expid}, expid + 1 if exist

        os.makedirs(self.logdir, exist_ok=True)

        if not self.eval:
            # create experiment directory
            self.experiment_id = self.get_expid(self.logdir, self.prefix)

            self.logdir = os.path.join(self.logdir, str(self.experiment_id))

            # create sub directory
            os.makedirs(os.path.join(self.logdir, 'models'), exist_ok=True)

            # write config
            json.dump(self.args, open(os.path.join(self.logdir, 'config.json'), 'w'), indent=4)
        else:
            self.experiment_id = self.args['exp_id']
            self.logdir = os.path.join(self.logdir, str(self.args['exp_id']))
            path = os.path.join(self.logdir, 'models', 'best.pth')

            # check experiment exists
            if not os.path.exists(path):
                print(f'Warning: No such Experiment -> {path}')
            else:
                self.load_model('best.pth')

            self.model = self.model.to(self.device)

    def save_model(self, filename, model=None):
        if model is None:
            model = self.model

        torch.save(model.cpu().state_dict(), os.path.join(self.logdir, f'models/{filename}'))
        model.to(self.device)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(os.path.join(self.logdir, f'models/{filename}')))

    def save_last_model(self, model=None):
        self.save_model('last.pth', model)

    def training(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def flush_history(self, history_file, first):
        if len(self.buffer) != 0:
            columns = sorted(self.buffer[0].keys())
            with open(history_file, 'a') as file:
                writer = csv.writer(file, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)
                if first:
                    writer.writerow(columns)

                for data in self.buffer:
                    writer.writerow(list(map(lambda x: data[x], columns)))

            self.buffer.clear()

    def append_history(self, history_file, data, first=False):  # row by row
        self.buffer.append(data)

        if len(self.buffer) >= self.save_history_interval:
            self.flush_history(history_file, first)
