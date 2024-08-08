import torch
import os
import logging

class EarlyStopping:
    def __init__(self, patience = 5, verbose = False, delta = 0, mode = 'min', save_path = 'checkpoints'):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation metric improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            mode (str): One of {'min', 'max'}. In 'min' mode, training will stop when the
                        quantity monitored has stopped decreasing; in 'max' mode it will
                        stop when the quantity monitored has stopped increasing.
                            Default: 'min'
            save_path (str): Directory where model checkpoints will be saved.
                            Default: 'checkpoints'
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_path = save_path
        self.logger = logging.getLogger(self.__class__.__name__)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if mode == 'min':
            self.val_metric_min = float('inf')
        elif mode == 'max':
            self.val_metric_max = float('-inf')
        else:
            raise ValueError("mode must be 'min' or 'max'")

    def __call__(self, val_metric, model):
        if self.mode == 'min':
            score = -val_metric
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_metric, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_metric, model)
                self.counter = 0
        elif self.mode == 'max':
            score = val_metric
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_metric, model)
            elif score < self.best_score - self.delta:
                self.counter += 1
                if self.verbose:
                    self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_metric, model)
                self.counter = 0
        


    def save_checkpoint(self, val_metric, model):
        '''Saves model when validation metric improves.'''
        if self.verbose:
            if self.mode == 'min':
                self.logger.info(f'Validation metric decreased ({self.val_metric_min:.6f} --> {val_metric:.6f}).  Saving model ...')
                self.val_metric_min = val_metric
            elif self.mode == 'max':
                self.logger.info(f'Validation metric increased ({self.val_metric_max:.6f} --> {val_metric:.6f}).  Saving model ...')
                self.val_metric_max = val_metric
        torch.save(model.state_dict(), os.path.join(self.save_path, 'checkpoint.pth'))