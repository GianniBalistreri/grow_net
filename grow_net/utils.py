"""

Utility functions

"""

import numpy as np
import os

from datetime import datetime
from typing import List


def remove_single_queries(query: np.array, target: np.array) -> List[str]:
    """
    Remove single queries used for pairwise ranking model

    :param query: np.array
        Query containing id's

    :param target: np.array
        Target values

    :return: List[str]
        Non-single id's
    """
    _idx: List[str] = []
    _unique_idx: np.array = np.unique(query)
    for i in range(0, len(_unique_idx), 1):
        _id: np.array = np.where(query == _unique_idx[i])[0]
        if len(_id) > 1 and len(np.unique(target[_id])) > 1:
            _idx.append(_id)
    return np.concatenate(_idx).ravel().tolist()


class Log:
    """
    Class for handling logging
    """
    def __init__(self,
                 write: bool = False,
                 level: str = 'info',
                 env: str = 'dev',
                 logger_file_path: str = None
                 ):
        """
        :param write: bool
            Write logging file or not

        :param level: str
            Name of the logging level of the messge
                -> info: Logs any information
                -> warn: Logs warnings
                -> error: Logs errors including critical messages

        :param env: str
            Name of the logging environment to use
                -> dev: Development - Logs any information
                -> stage: Staging - Logs only warnings and errors including critical messages
                -> prod: Production - Logs only errors including critical messages

        :param logger_file_path: str
            Complete file path of the logger file
        """
        self.write: bool = write
        self.timestamp_format: str = '%Y-%m-%d %H:%M:%S'
        self.msg: str = f'{datetime.now().strftime(self.timestamp_format)} | '
        if write:
            if logger_file_path is None:
                self.log_file_path: str = os.path.join(os.getcwd(), 'log.txt')
            else:
                self.log_file_path: str = logger_file_path
        else:
            self.log_file_path: str = None
        self.levels: List[str] = ['info', 'warn', 'error']
        _env: dict = dict(dev=0, stage=1, prod=2)
        if env in _env.keys():
            self.env: int = _env.get(env)
        else:
            self.env: int = _env.get('dev')
        if level in self.levels:
            self.level: int = self.levels.index(level)
        else:
            self.level: int = 0

    def _write(self):
        """
        Write log file
        """
        with open(file=self.log_file_path, mode='a', encoding='utf-8') as _log:
            _log.write('{}\n'.format(self.msg))

    def log(self, msg: str):
        """
        Log message

        :param msg: str
            Message to log
        """
        if self.level >= self.env:
            if self.level == 0:
                _level_description: str = ''
            elif self.level == 1:
                _level_description: str = 'WARNING: '
            elif self.level == 2:
                _level_description: str = 'ERROR: '
            self.msg = '{}{}'.format(self.msg, msg)
            if self.write:
                self._write()
            else:
                print(self.msg)
