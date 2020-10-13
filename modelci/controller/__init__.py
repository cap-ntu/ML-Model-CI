#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 6/29/2020
"""
import atexit

from modelci.controller.executor import JobExecutor

job_executor = JobExecutor()
job_executor.start()


@atexit.register
def terminate_controllers():
    job_executor.join()
    print(f'Exiting job executor.')


__all__ = ['job_executor']
