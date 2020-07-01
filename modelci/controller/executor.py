#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 6/29/2020
"""
from queue import Queue
from threading import Thread
from typing import Union

from docker.models.containers import Container

from modelci.hub.profiler import Profiler
from modelci.metrics.benchmark.metric import BaseModelInspector
from modelci.persistence.service import ModelService
from modelci.types.bo import ModelBO


class Job(object):
    def __init__(
            self,
            client: BaseModelInspector,
            device: str,
            model_info: ModelBO,
            container_name: str = None
    ):
        self.client = client
        self.device = device
        self.model = model_info
        self.container_name = container_name


class JobExecutor(Thread):
    _instance = None
    _queue_finish_flag = object()

    def __init__(self, q_size: int = 200):
        if self._instance is not None:
            return
        super().__init__()
        self.job_queue: Queue[Union[Job, object]] = Queue(maxsize=q_size)
        self._hold_container: Queue[Container] = Queue(maxsize=10)
        self._instance = self

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            return super().__new__(cls, *args, **kwargs)
        else:
            return cls._instance

    def __del__(self):
        for container in iter(self._hold_container.get, None):
            container.stop()

    def submit(self, job: Job):
        """Submit a profiling job to the executor."""
        self.job_queue.put(job)

    def finish(self):
        """The executor stops accepting new coming jobs.

        This function should be called before `join`. Otherwise, the executor will never stop.

        TODO: Save exit when there is an exception. Try excepthook in python 3.8
        """
        self.job_queue.put(self._queue_finish_flag)

    def run(self) -> None:
        from modelci.hub.deployer.dispatcher import serve

        for job in iter(self.job_queue.get, None):
            # exit the queue
            if job is self._queue_finish_flag:
                break
            # start a new container if container not started
            if job.container_name is None:
                container = serve(save_path=job.model.saved_path, device=job.device)
                container_name = container.name
                # remember to clean-up the created container
                self._hold_container.put(container)
            else:
                container_name = job.container_name
            profiler = Profiler(model_info=job.model, server_name=container_name, inspector=job.client)
            dpr = profiler.diagnose(device=job.device)
            ModelService.append_dynamic_profiling_result(job.model.id, dynamic_result=dpr)
            if job.container_name is None:
                # get holding container
                self._hold_container.get().stop()
