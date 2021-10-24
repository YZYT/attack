import time

import torch
import torch.nn.functional as F


from models.losses.sign_loss import SignLoss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


class Tester(object):
    def __init__(self, model, device, verbose=True):
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        self.model = model
        self.device = device
        self.verbose = verbose

    def test(self, dataloader, msg='Testing Result', compare=[]):
        self.model.eval()
        loss_meter = 0
        acc_meter = 0
        runcount = 0

        start_time = time.time()
        with torch.no_grad():
            for i, load in enumerate(dataloader):
                data, target = load[:2]
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                pred = self.model(data)
                loss_meter += F.cross_entropy(pred, target, reduction='sum').item()  # sum up batch loss
                pred = pred.max(1, keepdim=True)[1]  # get the index of the max log-probability
                compare.append((pred, target))
                acc_meter += pred.eq(target.view_as(pred)).sum().item()
                runcount += data.size(0)
                if self.verbose:
                    print(f'{msg} [{i + 1}/{len(dataloader)}]: '
                          f'Loss: {loss_meter / runcount:6.4f} '
                          f'Acc: {100 * acc_meter / runcount:6.2f} ({time.time() - start_time:.2f}s)', end='\r')

        loss_meter /= runcount
        acc_meter = 100 * acc_meter / runcount

        if self.verbose:
            print(f'{msg}: '
                  f'Loss: {loss_meter:6.4f} '
                  f'Acc: {acc_meter:6.2f} ({time.time() - start_time:.2f}s)')
            print()

        return {'loss': loss_meter, 'acc': acc_meter, 'time': time.time() - start_time}


class Trainer(object):
    def __init__(self, model, optimizer, scheduler, device):
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        self.models = model
        self.optimizers = optimizer
        self.schedulers = scheduler
        self.device = device
        self.steps = 0


    def Fed_avg(self):

        def get_weight(optimizer):
            private = True
            for param_group in optimizer.param_groups:
                if private:
                    private = False
                    continue
                for p in param_group['params']:
                    yield p.data


        weights_iters = [get_weight(optimizer) for optimizer in self.optimizers]
        averages = []
        with torch.no_grad():
            # steps = len(weights_iters[0])
            while True:
                a = None
                end = False
                for weights_iter in weights_iters:
                    try:
                        weights = next(weights_iter)
                        if a is None:
                            a = torch.zeros_like(weights)
                        a.add_(weights, alpha=1.0 / len(weights_iters))

                    except StopIteration:
                        end = True
                        break
                
                if end:
                    break

                averages.append(a)
 

            for optimizer in self.optimizers:
                private = True
                i = 0
                for param_group in optimizer.param_groups:
                    if private:
                        private = False
                        continue

                    for p in param_group['params']:
                        p.data.copy_(averages[i])
                        i += 1


    def train(self, e, train_datas):
        iters = list(map(lambda loader: iter(loader), train_datas))

        while True:
            end = False
            for model, optimizer, data_iter in zip(self.models, self.optimizers, iters):
                model.train()
                loss_meter = 0
                acc_meter = 0

                start_time = time.time()
                try:
                    data, target = next(data_iter)
                except StopIteration:
                    end = True
                    break

                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                pred = model(data)
                loss = F.cross_entropy(pred, target)

                loss.backward()
                optimizer.step()


            if end:
                break
            
            # self.steps += 1
            # if self.steps == 10:
            #     self.steps = 0
            #     self.Fed_avg()

                # loss_meter += loss.item()
                # acc_meter += accuracy(pred, target)[0].item()

                    # print(f'Epoch {e:3d} [{i:4d}/{len(dataloader):4d}] '
                    #     f'Loss: {loss_meter / (i + 1):6.4f} '
                    #     f'Acc: {acc_meter / (i + 1):.4f} ({time.time() - start_time:.2f}s)', end='\r')

                # print()


                # loss_meter /= len(dataloader)
                # acc_meter /= len(dataloader)

        for scheduler in self.schedulers:
            if scheduler is not None:
                scheduler.step()

            # return {'loss': loss_meter,
            #         'acc': acc_meter,
            #         'time': time.time() - start_time}


    def train_one(self, e, dataloader):
        model, optimizer, scheduler = self.models[0], self.optimizers[0], self.schedulers[0]
        model.train()
        loss_meter = 0
        acc_meter = 0

        start_time = time.time()
        for i, (data, target) in enumerate(dataloader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            optimizer.zero_grad()

            pred = model(data)
            loss = F.cross_entropy(pred, target)
            # loss_fn = torch.nn.MSELoss(reduction='mean')
            # loss = loss_fn(pred.reshape(target.shape), target)

            loss.backward()
            optimizer.step()

            loss_meter += loss.item()
            acc_meter += accuracy(pred, target)[0].item()

            print(f'Epoch {e:3d} [{i:4d}/{len(dataloader):4d}] '
                  f'Loss: {loss_meter / (i + 1):6.4f} '
                  f'Acc: {acc_meter / (i + 1):.4f} ({time.time() - start_time:.2f}s)', end='\r')

        print()

        loss_meter /= len(dataloader)
        acc_meter /= len(dataloader)

        if scheduler is not None:
            scheduler.step()

        return {'loss': loss_meter,
                'acc': acc_meter,
                'time': time.time() - start_time}


    def test(self, dataloader, msg='Testing Result'):
        model = self.models[0]
        model.eval()
        loss_meter = 0
        acc_meter = 0
        runcount = 0

        start_time = time.time()
        with torch.no_grad():
            for i, load in enumerate(dataloader):
                data, target = load[:2]
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                pred = model(data)
                loss_meter += F.cross_entropy(pred, target, reduction='sum').item()
                # loss_meter += F.mse_loss(pred.reshape(target.shape), target, reduction='sum').item()
                pred = pred.max(1, keepdim=True)[1]

                acc_meter += pred.eq(target.view_as(pred)).sum().item()
                runcount += data.size(0)
                print(f'{msg} [{i + 1}/{len(dataloader)}]: '
                      f'Loss: {loss_meter / runcount:6.4f} '
                      f'Acc: {acc_meter / runcount:6.2f} ({time.time() - start_time:.2f}s)', end='\r')

        loss_meter /= runcount
        acc_meter = 100 * acc_meter / runcount
        print(f'{msg}: '
              f'Loss: {loss_meter:6.4f} '
              f'Acc: {acc_meter:6.2f} ({time.time() - start_time:.2f}s)')
        print()

        return {'loss': loss_meter,
                'acc': acc_meter,
                'time': time.time() - start_time}
