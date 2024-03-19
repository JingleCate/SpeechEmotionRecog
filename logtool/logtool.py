import datetime
import os
import sys

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
            print("💻 [Message]   ⏩", message, end="⏪\n")
         
            print("✂ ", '-' * (terminal_legnth - 2), sep='')
            ret = func(*args, **kwargs)
            print("✂ ", '-' * (terminal_legnth - 2), sep='', end='')
            return ret
        return wrapper
    return _log
