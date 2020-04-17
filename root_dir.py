#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/12/13
"""

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 存储项目所在的绝对路径

PTH_DIR = os.path.join(ROOT_DIR, 'checkpoint', 'snapshot')
GAZE_MODEL = os.path.join(ROOT_DIR, 'checkpoint', 'gaze')

DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMGS_DIR = os.path.join(DATA_DIR, 'imgs')
