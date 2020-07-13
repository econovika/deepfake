from extract.data_loader import WiderDataset, collate_fn
from extract.detect.process_yolo import Process
from extract.detect.yolo_face import YoloLoss
from torch import manual_seed, no_grad, stack
from torch.utils.data import DataLoader
from torch.optim import SGD

import os
import torch.multiprocessing as mp


class TrainEvalModel:
    def __init__(
            self,
            model, num_classes, valid_anchors_wh,
            input_data_paths,
            train_batch_size=1, eval_batch_size=1,
            lr=0.01, momentum=0.8, num_epochs=10,
            log_interval=1, num_workers=1,
            cuda=True,
            seed=1
    ):
        self.cuda = cuda

        self.model = model

        self.num_classes, self.valid_anchors_wh = num_classes, valid_anchors_wh

        self.input_data_paths = input_data_paths

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.lr, self.momentum, self.num_epochs = lr, momentum, num_epochs

        self.log_interval, self.num_workers = log_interval, num_workers
        self.seed = seed

    def __init_dataloader(self, data_paths, kwargs, rank=0):
        """
        Initialize train or eval dataloader due to data_paths param
        """
        manual_seed(self.seed + rank)
        file_path, input_dir = data_paths
        dataset = WiderDataset(file_path, input_dir)
        dataloader = DataLoader(dataset, **kwargs)
        return dataloader

    @staticmethod
    def __log_print(params):
        pid = params.get('pid', None)
        mode = params.get('mode', None)
        epoch = params.get('epoch', None)
        processed_size = params.get('processed_size', None)
        input_size = params.get('input_size', None)
        loss = params.get('loss', None)

        msg = (
            f'PID: {pid}\t{mode.upper()} '
            f'EPOCH: {epoch} [{processed_size}/{input_size} ({(processed_size / input_size * 100):.0f}%)]\t'
            f'LOSS: {loss:.6f}'
        )
        print(msg)

    def train_step(self, epoch, rank):
        self.model.train()
        kwargs = {
            'batch_size': self.train_batch_size,
            'num_workers': 1,
            'shuffle': True,
            'collate_fn': collate_fn
        }
        dataloader = self.__init_dataloader(
            self.input_data_paths['train'],
            kwargs,
            rank
        )
        if self.cuda:
            # Move model to GPU before initializing the optimizer as discussed at:
            # https://discuss.pytorch.org/t/effect-of-calling-model-cuda-after-constructing-an-optimizer/15165/7
            self.model.cuda()
        optimizer = SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum
        )
        loss = YoloLoss(self.num_classes, self.valid_anchors_wh)

        for batch_idx, (X, y_true) in enumerate(dataloader, 1):
            optimizer.zero_grad()
            if self.cuda:
                X.cuda()
            y_pred = self.model(X)
            loss = loss(y_pred, y_true)
            loss.backward()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                params = {
                    'mode': 'train',
                    'epoch': epoch,
                    'processed_size': batch_idx * self.train_batch_size,
                    'dataset_size': len(dataloader.dataset),
                    'loss': loss.item()
                }
                if not self.cuda:
                    pid = os.getpid()
                    params['pid'] = pid
                self.__log_print(params=params)

    def eval_step(self):
        self.model.eval()
        kwargs = {
            'batch_size': self.eval_batch_size,
            'num_workers': 1,
            'shuffle': False,
            'collate_fn': collate_fn
        }
        dataloader = self.__init_dataloader(
            self.input_data_paths['eval'],
            kwargs,
        )
        if self.cuda:
            self.model.cuda()
        loss = YoloLoss(self.num_classes, self.valid_anchors_wh)

        preds, avg_loss = [], 0
        with no_grad():
            for batch_idx, (X, y_true) in enumerate(dataloader, 1):
                if self.cuda:
                    X.cuda()
                y_pred = self.model(X)
                batch_pred = Process(
                    y_pred=y_pred,
                    num_classes=self.num_classes,
                    valid_anchors_wh=self.valid_anchors_wh,
                    GIoU_thresh=0.7,
                    score_thresh=0.5
                )
                loss = loss(y_pred, y_true)
                avg_loss += loss.item()
                preds.append(batch_pred)
        params = {
            'mode': 'test',
            'dataset_size': len(dataloader.dataset),
            'loss': avg_loss
        }
        self.__log_print(params=params)
        return stack(preds, dim=2).squeeze(dim=0)

    def __call__(self, *args, **kwargs):
        """
        Implement Hogwild multiprocessing training or evaluation of given model
        based on pytorch examples:
        https://github.com/pytorch/examples/blob/master/mnist_hogwild/train.py

        Or, if not cpu but cuda, just perform single-process
        training or evaluation on GPU
        """
        is_train = kwargs['train']
        if not self.cuda:
            if is_train:
                mp.set_start_method('spawn')
                self.model.share_memory()
                processes = []
                for epoch in range(1, self.num_epochs + 1):
                    for rank in range(self.num_workers):
                        p = mp.Process(target=self.train_step, args=(epoch, rank))
                        p.start()
                        processes.append(p)
                    for p in processes:
                        p.join()
                    # Perform model evaluation after all processes complete one epoch
                    self.eval_step()
            else:
                self.eval_step()
        else:
            self.model.cuda()
            if is_train:
                for epoch in range(1, self.num_epochs + 1):
                    self.train_step(epoch, rank=0)
                    self.eval_step()
            else:
                self.eval_step()
