#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 1/12/2021
"""
import abc
import threading
from concurrent.futures import Future
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Optional

import pytorch_lightning as pl


class BaseTrainer(abc.ABC):
    """Trainer interface."""

    def start(self):
        raise NotImplementedError('Method `start` not implemented.')

    def join(self, timeout=None):
        """Wait for the trainer to finish training."""

        raise NotImplementedError('Method `join` not implemented.')

    def terminate(self):
        raise NotImplementedError('Method `terminate` not implemented.')

    def resume_soon(self):
        """Resume from a pause."""

        raise NotImplementedError('Method `resume_soon` not implemented.')

    def pause_soon(self):
        """Pause training for a while"""
        raise NotImplementedError('Method `pause_soon` not implemented.')

    def export_model(self):
        raise NotImplementedError('Method `export_model` not implemented.')

    def set_device(self):
        raise NotImplementedError('Method `set_device` not implemented.')


class PyTorchTrainer(BaseTrainer):
    """
    PyTorch Trainer inherited from `pytorch_lighting`.
    """

    def __init__(self, model: pl.LightningModule, data_loader_kwargs: dict = None, trainer_kwargs: dict = None):
        self.model = model
        trainer_kwargs = trainer_kwargs or dict()
        self.trainer_engine = pl.Trainer(**trainer_kwargs)
        self._data_loader_kwargs = data_loader_kwargs or dict()

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._event_pause = threading.Event()
        self._task: Optional[Future] = None

    def start(self):
        self._task = self._executor.submit(self.trainer_engine.fit, self.model, **self._data_loader_kwargs)

    def join(self, timeout=None):
        if self._task:
            self._task.result(timeout=timeout)

    def terminate(self):
        if self._task:
            # trigger pytorch lighting training graceful shutdown via a ^C
            self._task.set_exception(KeyboardInterrupt())

    def export_model(self):
        return self.model.cpu()

    def resume_soon(self):
        if self._event_pause.is_set():
            self._event_pause.clear()
            return True
        return False

    def pause_soon(self):
        if not self._event_pause.is_set():
            self._event_pause.set()
            return True
        return False

    def set_device(self):
        pass
