import torch
import random
import numpy as np
import logging
import os
import time
import sys


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_optimizer(args, model):
    if args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    if args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=args.lr)
    if args.optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=args.lr)


def get_scheduler(args, optimizer):
    if args.sched == 'steplr':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if args.sched == 'cal':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=args.lr*1e-3)
    if args.sched == 'car':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=args.lr*1e-3)
    if args.sched == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


def ensure_dir(path):
    if not os.path.isdir(path):
        try:
            sleeptime = random.randint(0, 3)
            time.sleep(sleeptime)
            os.makedirs(path)
        except:
            print('conflict !!!')

_default_level_name = os.getenv('ENGINE_LOGGING_LEVEL', 'INFO')
_default_level = logging.getLevelName(_default_level_name.upper())


class LogFormatter(logging.Formatter):
    log_fout = None
    date_full = '[%(asctime)s %(lineno)d@%(filename)s:%(name)s] '
    date = '%(asctime)s '
    msg = '%(message)s'

    def format(self, record):
        if record.levelno == logging.DEBUG:
            mcl, mtxt = self._color_dbg, 'DBG'
        elif record.levelno == logging.WARNING:
            mcl, mtxt = self._color_warn, 'WRN'
        elif record.levelno == logging.ERROR:
            mcl, mtxt = self._color_err, 'ERR'
        else:
            mcl, mtxt = self._color_normal, ''

        if mtxt:
            mtxt += ' '

        if self.log_fout:
            self.__set_fmt(self.date_full + mtxt + self.msg)
            formatted = super(LogFormatter, self).format(record)
            return formatted

        self.__set_fmt(self._color_date(self.date) + mcl(mtxt + self.msg))
        formatted = super(LogFormatter, self).format(record)

        return formatted

    if sys.version_info.major < 3:
        def __set_fmt(self, fmt):
            self._fmt = fmt
    else:
        def __set_fmt(self, fmt):
            self._style._fmt = fmt

    @staticmethod
    def _color_dbg(msg):
        return '\x1b[36m{}\x1b[0m'.format(msg)

    @staticmethod
    def _color_warn(msg):
        return '\x1b[1;31m{}\x1b[0m'.format(msg)

    @staticmethod
    def _color_err(msg):
        return '\x1b[1;4;31m{}\x1b[0m'.format(msg)

    @staticmethod
    def _color_omitted(msg):
        return '\x1b[35m{}\x1b[0m'.format(msg)

    @staticmethod
    def _color_normal(msg):
        return msg

    @staticmethod
    def _color_date(msg):
        return '\x1b[32m{}\x1b[0m'.format(msg)


def get_logger(log_dir=None, log_file=None, stout=False, formatter=LogFormatter):
    logger = logging.getLogger()
    logger.setLevel(_default_level)
    del logger.handlers[:]

    if log_dir and log_file:
        log_file = log_dir + log_file
        ensure_dir(log_dir)
        LogFormatter.log_fout = True
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter(datefmt='%d %H:%M:%S'))
        logger.addHandler(file_handler)
    if stout:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter(datefmt='%d %H:%M:%S'))
        stream_handler.setLevel(0)
        logger.addHandler(stream_handler)
    return logger


