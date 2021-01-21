#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 1/12/2021
"""
import abc
from concurrent.futures.thread import ThreadPoolExecutor

import pytorch_lightning as pl


class BaseTrainer(abc.ABC):
    """Trainer interface."""

    def start(self):
        raise NotImplementedError('Method `start` not implemented.')

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
    def __init__(self, model, data_loader_kwargs, trainer_kwargs):
        self.model = model
        self.trainer_engine = pl.Trainer(**trainer_kwargs)
        self._data_loader_kwargs = data_loader_kwargs

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._task = None

    def start(self):
        self._task = self._executor.submit(self.trainer_engine.fit, self.model, **self._data_loader_kwargs)

    def terminate(self):
        pass

    def export_model(self):
        pass

    def resume_soon(self):
        pass

    def pause_soon(self):
        pass

    def set_device(self):
        pass
