from logging import FileHandler
import os


class MakeDirsFileHandler(FileHandler):
    """
    Custom file handler used for logging, so it creates the directory that contains the log file automatically
    when initializing.
    """

    def __init__(self, filename: str, mode='a', encoding=None, delay=0):
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(os.path.dirname(filename))
        super().__init__(filename, mode, encoding, delay)
