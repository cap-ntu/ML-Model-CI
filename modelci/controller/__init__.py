#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yli056@e.ntu.edu.sg
Date: 6/29/2020
"""
from modelci.controller.executor import JobExecutor

job_executor = JobExecutor()
job_executor.start()

__all__ = ['job_executor']
