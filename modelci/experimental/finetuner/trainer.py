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
import torch

import modelci.experimental.curd.model_train as model_train_curd
from modelci.experimental.finetuner import OUTPUT_DIR
from modelci.experimental.finetuner.pytorch_datamodule import PyTorchDataModule
from modelci.experimental.finetuner.transfer_learning import freeze, FineTuneModule
from modelci.experimental.model.model_train import TrainingJob, TrainingJobUpdate
from modelci.hub.cache_manager import get_remote_model_weight
from modelci.persistence.service_ import get_by_id, update_model
from modelci.types.models import ModelUpdateSchema
from modelci.types.models.common import Engine, ModelStatus, Status


class BaseTrainer(abc.ABC):
    """Trainer interface."""

    def __init__(self, id):
        self.id = id

    @classmethod
    @abc.abstractmethod
    def from_training_job(cls, training_job: TrainingJob):
        raise NotImplementedError('Method `from_training_job` not implemented.')

    @abc.abstractmethod
    def start(self):
        raise NotImplementedError('Method `start` not implemented.')

    @abc.abstractmethod
    def join(self, timeout=None):
        """Wait for the trainer to finish training."""

        raise NotImplementedError('Method `join` not implemented.')

    @abc.abstractmethod
    def terminate(self):
        raise NotImplementedError('Method `terminate` not implemented.')

    @abc.abstractmethod
    def resume_soon(self):
        """Resume from a pause."""

        raise NotImplementedError('Method `resume_soon` not implemented.')

    @abc.abstractmethod
    def pause_soon(self):
        """Pause training for a while"""
        raise NotImplementedError('Method `pause_soon` not implemented.')

    @abc.abstractmethod
    def export_model(self):
        raise NotImplementedError('Method `export_model` not implemented.')

    @abc.abstractmethod
    def set_device(self):
        raise NotImplementedError('Method `set_device` not implemented.')


class PyTorchTrainer(BaseTrainer):
    """
    PyTorch Trainer utilize :class:`pytorch_lighting.Trainer` as the engine.
    """

    def __init__(
            self,
            model: pl.LightningModule,
            model_id: str,
            id: str = None,
            data_loader_kwargs: dict = None,
            trainer_kwargs: dict = None,
    ):
        super().__init__(id)
        self.model = model
        trainer_kwargs = trainer_kwargs or dict()
        self.model_id = model_id
        self.trainer_engine = pl.Trainer(**trainer_kwargs)
        self._data_loader_kwargs = data_loader_kwargs or dict()

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._event_pause = threading.Event()
        self._task: Optional[Future] = None

    @classmethod
    def from_training_job(cls, training_job: TrainingJob) -> 'PyTorchTrainer':
        # TODO: only support fine-tune

        ml_model = get_by_id(training_job.model)
        if ml_model.engine != Engine.NONE:
            raise ValueError(f'Model engine expected `{Engine.NONE}`, but got {ml_model.engine}.')

        # download local cache
        cache_path = get_remote_model_weight(ml_model)
        net = torch.load(cache_path)
        freeze(module=net, n=-1, train_bn=True)

        # build pytorch lightning module
        fine_tune_module_kwargs = {
            'net': net,
            'loss': eval(str(training_job.loss_function))(),  # nosec
            'batch_size': training_job.data_module.batch_size,
            'num_workers': training_job.data_module.num_workers,
        }
        if training_job.optimizer_property.lr:
            fine_tune_module_kwargs['lr'] = training_job.optimizer_property.lr
        if training_job.lr_scheduler_property.gamma:
            fine_tune_module_kwargs['lr_scheduler_gamma'] = training_job.lr_scheduler_property.gamma
        if training_job.lr_scheduler_property.step_size:
            fine_tune_module_kwargs['step_size'] = training_job.lr_scheduler_property.step_size
        model = FineTuneModule(**fine_tune_module_kwargs)
        data_module = PyTorchDataModule(**training_job.data_module.dict(exclude_none=True))

        trainer_kwargs = training_job.dict(exclude_none=True, include={'min_epochs', 'max_epochs'})
        trainer = cls(
            id=training_job.id,
            model=model,
            data_loader_kwargs={'datamodule': data_module},
            trainer_kwargs={
                'default_root_dir': training_job.data_module.data_dir or OUTPUT_DIR,
                'weights_summary': None,
                'progress_bar_refresh_rate': 1,
                'num_sanity_val_steps': 0,
                'gpus': 1,  # TODO: set GPU number
                **trainer_kwargs,
            },
            model_id=training_job.model
        )
        return trainer

    def start(self):
        def training_done_callback(future):
            model_train_curd.update(TrainingJobUpdate(_id=self.id, status=Status.Pass))
            # TODO: save to database and update model_status, engine
            print(self.export_model())

        self._task = self._executor.submit(self.trainer_engine.fit, self.model, **self._data_loader_kwargs)
        self._task.add_done_callback(training_done_callback)
        model_train_curd.update(TrainingJobUpdate(_id=self.id, status=Status.Running))

        ml_model = get_by_id(self.model_id)
        if ModelStatus.DRAFT in ml_model.model_status:
            ml_model.model_status.remove(ModelStatus.DRAFT)
        ml_model.model_status.append(ModelStatus.TRAINING)
        update_model(str(ml_model.id), ModelUpdateSchema(model_status=ml_model.model_status))

    def join(self, timeout=None):
        if self._task:
            self._task.result(timeout=timeout)

    def terminate(self):
        if self._task:
            # trigger pytorch lighting training graceful shutdown via a ^C
            self._task.set_exception(KeyboardInterrupt())
            model_train_curd.update(TrainingJobUpdate(_id=self.id, status=Status.FAIL))
            ml_model = get_by_id(self.model_id)
            ml_model.model_status.remove(ModelStatus.TRAINING)
            ml_model.model_status.append(ModelStatus.DRAFT)
            update_model(str(ml_model.id), ModelUpdateSchema(model_status=ml_model.model_status))

    def export_model(self):
        return self.model.net.cpu()

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
