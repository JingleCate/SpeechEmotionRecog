import datetime
import os
import sys
import logging

from functools import wraps

terminal_legnth = os.get_terminal_size().columns
# print(terminal_legnth)

icons = {
    'error': "🚨 [ERROR]",
    'warning': "⌛ [WARNING]",
    'info': "🔋 [INFO]"
}


def log(T: str, message: str = ''):
    """Outputs a log message.

    Args:
        T (str): "error" or "warning" or "info".
        message (str): message to be outputted.
    """
    # 获取被调用函数所在模块文件名
    file = sys._getframe(1).f_code.co_filename
    def _log(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            time = datetime.datetime.now()
            print(icons[T], time, " @file[" + file + "] ", " @method[" + func.__name__ + "]", sep=' ')
            print("💻 [Message]   ⏩⏩", message, end="\n")
         
            print("✂ ", '-' * (terminal_legnth - 2), sep='')
            ret = func(*args, **kwargs)
            print("✂ ", '-' * (terminal_legnth - 2), sep='', end='')
            return ret
        return wrapper
    return _log


class myLogger:
    def __init__(self, 
                 log_dir: str=".",
                 filename: str=None,
                 log_level: int = logging.INFO
                 ):
        ###############################     logging config start       #################################
        self.log_dir = log_dir

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.log_filename = filename

        if not self.log_filename.endswith(".log"):
            self.log_filename = self.log_filename.split('.')[0] + ".log"

        self.log_level = log_level
        # LOG_FORMAT = "%(asctime)s - [%(levelname)s] - (in %(filename)s:%(lineno)d, %(funcName)s()) \t⏩⏩ %(message)s"
        self.LOG_FORMAT = "%(asctime)s - [%(levelname)s] - ⏩⏩ %(message)s"
        self.DATE_FORMAT = "%Y/%m/%d %H:%M:%S"

        self.formatter = logging.Formatter(fmt=self.LOG_FORMAT, datefmt=self.DATE_FORMAT)  #self.formatter

        # file handler
        self.fhandler = logging.FileHandler(filename=os.path.join(self.log_dir, self.log_filename), encoding='utf-8')
        self.fhandler.setLevel(self.log_level)
        self.fhandler.setFormatter(self.formatter)

        # control platform handler
        self.chandler = logging.StreamHandler()
        self.chandler.setLevel(self.log_level)
        self.chandler.setFormatter(self.formatter)
        ###############################     logging config end        #################################
    
    @property
    def get_logger(self):
        """Instead of using function calling style, use property style(just not with '()').

        Returns
        -------
        logger
        """
        logger = logging.getLogger(self.log_filename.split('.')[0])        # get a logger by the name.
        logger.setLevel(self.log_level)
        logger.addHandler(self.fhandler)
        logger.addHandler(self.chandler)

        return logger

