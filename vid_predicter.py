#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/4/21
"""
import os
import time

import cv2

from gaze_predicter import GazePredicter
from root_dir import VIDS_DIR
from utils.project_utils import mkdir_if_not_exist, get_current_time_str
from utils.video_utils import init_vid, write_video


class VidPredicter(object):
    def __init__(self):
        self.gp = GazePredicter()  # 目光预测
        self.frames_dir = os.path.join(VIDS_DIR, 'frames')
        self.out_path = os.path.join(VIDS_DIR, 'out.{}.mp4'.format(get_current_time_str()))
        mkdir_if_not_exist(self.frames_dir)

    def predict_path(self, path):
        print('[Info] 视频路径: {}'.format(path))
        cap, n_frame, fps, h, w = init_vid(path)
        n_frame = 100
        # img_list = []
        for i in range(0, n_frame):
            print('-' * 50)
            s_time = time.time()
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            try:
                face_dict = self.gp.predict(frame)
                img_draw = face_dict['img_draw']
                # img_list.append(img_draw)
            except Exception as e:
                continue

            img_path = os.path.join(self.frames_dir, '{}.jpg'.format(str(i).zfill(4)))
            cv2.imwrite(img_path, img_draw)
            print('[Info] 帧预测完成: {} / {}'.format(str(i), time.time() - s_time))


def main():
    vid_path = os.path.join(VIDS_DIR, 'vid_no_glasses.mp4')
    vp = VidPredicter()
    vp.predict_path(vid_path)


if __name__ == '__main__':
    main()
