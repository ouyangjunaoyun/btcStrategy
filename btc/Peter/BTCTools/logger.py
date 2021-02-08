import logging
from logging import FileHandler

import colorlog

loggers = {}


def get_logger(name='backtester', output_path=None):
    global loggers
    if name in loggers:
        return loggers.get(name)

    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"))
    handler.setLevel(logging.ERROR)

    loggers[name] = colorlog.getLogger(name)
    loggers[name].setLevel(logging.DEBUG)
    loggers[name].addHandler(handler)

    if output_path is not None:
        fileHandler = FileHandler(str(output_path))
        loggers[name].addHandler(fileHandler)

    return loggers[name]
