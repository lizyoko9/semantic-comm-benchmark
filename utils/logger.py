"""Logging wrapper for TensorBoard and console output."""

import os
import sys
import logging
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Combined TensorBoard and console logger."""

    def __init__(self, log_dir: str, name: str = "semantic_comm"):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        self.console = logging.getLogger(name)
        self.console.setLevel(logging.INFO)
        if not self.console.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter("[%(asctime)s %(levelname)s] %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
            )
            self.console.addHandler(handler)

    def info(self, msg: str):
        self.console.info(msg)

    def scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def image(self, tag: str, img_tensor, step: int):
        self.writer.add_image(tag, img_tensor, step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()
