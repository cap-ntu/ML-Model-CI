#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 2/2/2021
"""
from typing import Dict

from modelci.experimental.finetuner.trainer import BaseTrainer


class Coordinator(object):

    def __init__(self):
        self.pool: Dict[str, BaseTrainer] = dict()

    def get_job_by_id(self, job_id: str):
        return self.pool.get(job_id, None)

    def submit_job(self, trainer: BaseTrainer):
        self.pool[trainer.id] = trainer
        trainer.start()

    def delete_job_by_id(self, job_id: str):
        trainer = self.pool.pop(job_id, None)
        if trainer is not None:
            trainer.terminate()

    def delete_all(self):
        for trainer in self.pool.values():
            trainer.terminate()
