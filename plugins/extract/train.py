from extract.data_loader import WiderDataset, collate_fn
from extract.detect.utils import iou_and_generalized_iou
from extract.detect.process_yolo import Process
from extract.detect.yolo_face import YoloLoss
from torch import manual_seed, no_grad, stack
from torch.utils.data import DataLoader
from datetime import datetime
from torch.optim import SGD

import os
import torch.multiprocessing as mp


class TrainEvalModel:
    def __init__(
            self,
            model,
            input_data_paths,
            train_batch_size=1, eval_batch_size=1,
            lr=0.01, momentum=0.8, num_epochs=10,
            log_interval=1, num_workers=1, max_patience=5,
            model_save_dir=None, early_stopping=10,
            cuda=True,
            seed=1,
            valid_anchors_wh=None, num_classes=2
    ):
        """
        :param model: Pytorch yolo_face.YoloFaceNetwork model example
        :param num_classes: Number of predicted classes by model (default: 2)
        :param input_data_paths:
        :param train_batch_size:
        :param eval_batch_size:
        :param lr:
        :param momentum:
        :param num_epochs:
        :param log_interval:
        :param num_workers:
        :param max_patience:
        :param model_save_dir:
        :param early_stopping:
        :param cuda:
        :param seed:
        :param valid_anchors_wh:
        """
        self.cuda = cuda

        self.model = model

        self.num_classes, self.valid_anchors_wh = num_classes, valid_anchors_wh

        self.input_data_paths = input_data_paths

        self.train_batch_size, self.eval_batch_size = train_batch_size, eval_batch_size

        self.lr, self.momentum, self.num_epochs = lr, momentum, num_epochs

        self.log_interval, self.num_workers, self.model_save_dir = log_interval, num_workers, model_save_dir
        self.max_patience, self.early_stopping = max_patience, early_stopping
        self.seed = seed

        self.best_val_loss, self.cur_patience = float('inf'), 0

    def __init_dataloader(self, data_paths, kwargs, rank=0):
        """
        Initialize train or eval dataloader due to data_paths param,

        :param data_paths: str, path to train or eval dataset
        :param rank: int, current process (dataset parts between processes should be non-overlapping)
        """
        manual_seed(self.seed + rank)
        file_path, input_dir = data_paths
        dataset = WiderDataset(file_path, input_dir)
        dataloader = DataLoader(dataset, **kwargs)
        return dataloader

    @staticmethod
    def __log_print(params):
        """
        Display logging message (train / eval mode)
        """
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

    def __save_model(self, epoch, loss):
        """
        Save current best model to predefined directory or default: ./models
        """
        if self.model_save_dir is None:
            if not os.path.isdir('./models'):
                os.mkdir('./models')
            path = f"./models/model_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}_epoch_{epoch}_loss_{loss:.4f}"
        else:
            path = self.model_save_dir
        print("Model {0} saved".format(path))

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

    def eval_step(self, train_epoch=None):
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

        preds, GIoU_list, avg_loss = [], [], 0
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

                GIoU = iou_and_generalized_iou(batch_pred[0:4], y_true[0:4])
                GIoU_list.append(GIoU)
        params = {
            'mode': 'test',
            'dataset_size': len(dataloader.dataset),
            'loss': avg_loss
        }
        self.__log_print(params=params)

        # Save best model for validation mode
        if train_epoch is not None:
            if avg_loss < self.best_val_loss:
                self.__save_model(train_epoch, avg_loss)
                self.best_val_loss = avg_loss

                self.cur_patience = 0
            else:
                if self.cur_patience >= self.max_patience:
                    # Reduce on plateau learning schedule
                    # https://github.com/ethanyanjiali/deep-vision/blob/master/YOLO/tensorflow/train.py#L56
                    self.lr /= 10
                self.cur_patience += 1
        return stack(preds, dim=2).squeeze(dim=0), stack(GIoU_list, dim=2).squeeze(dim=0)

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
                    self.eval_step(train_epoch=epoch)
                    if self.cur_patience == self.early_stopping:
                        msg = f'Early stop at epoch: {epoch}, ' \
                              f'max repeated non improving limit is met: {self.cur_patience}'
                        print(msg)
                        break
            else:
                preds, GIoU_list = self.eval_step()
                return preds, GIoU_list
