#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 2/1/2021
"""
from typing import List

from modelci.experimental.model.model_train import TrainingJob


def exists_by_id(id: str) -> bool:
    raise NotImplementedError()


def get_by_id(id: str) -> TrainingJob:
    raise NotImplementedError()


def get_all() -> List[TrainingJob]:
    raise NotImplementedError()


def save(training_job: TrainingJob) -> int:
    raise NotImplementedError()


def update() -> int:
    raise NotImplementedError()


def delete_by_id(id: str) -> int:
    raise NotImplementedError()


def delete_all() -> int:
    raise NotImplementedError()
