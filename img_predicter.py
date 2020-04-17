#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/4/17
"""

import os

import cv2
import numpy as np
import torch
from torchvision import transforms

from models.pfld import PFLDInference
from mtcnn.detector import detect_faces
from root_dir import DATA_DIR, PTH_DIR, IMGS_DIR


class ImgPredicter(object):
    """
    图像预测
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_path = os.path.join(PTH_DIR, 'checkpoint.pth.tar')
        checkpoint = torch.load(model_path, map_location=self.device)

        plfd_backbone = PFLDInference().to(self.device)
        plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
        plfd_backbone.eval()

        self.plfd_backbone = plfd_backbone.to(self.device)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def predict_landmarks(self, img_op):
        height, width = img_op.shape[:2]
        bounding_boxes, landmarks = detect_faces(img_op)

        res_list = []

        for box in bounding_boxes:
            score = box[4]
            x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            size = int(max([w, h]) * 1.1)
            cx = x1 + w // 2
            cy = y1 + h // 2
            x1 = cx - size // 2
            x2 = x1 + size
            y1 = cy - size // 2
            y2 = y1 + size

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)

            cropped = img_op[y1:y2, x1:x2]
            if dx > 0 or dy > 0 or edx > 0 or edy > 0:
                cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

            cropped = cv2.resize(cropped, (112, 112))

            input = cv2.resize(cropped, (112, 112))
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            input = self.transform(input).unsqueeze(0).to(self.device)
            _, landmarks = self.plfd_backbone(input)

            pre_landmark = landmarks[0]
            pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [size, size]

            res_list.append([pre_landmark, x1, y1])  # 添加

        return res_list

    def predict(self, img_op):
        res_list = self.predict_landmarks(img_op)
        count = 0
        for res in res_list:
            pre_landmark, x1, y1 = res
            for (x, y) in pre_landmark.astype(np.int32):
                if count == 96 or count == 97:
                    cv2.circle(img_op, (x1 + x, y1 + y), 2, (0, 255, 255), -1)
                else:
                    cv2.circle(img_op, (x1 + x, y1 + y), 1, (0, 0, 255))
                count += 1
        return img_op

    def predict_path(self, img_path):
        img_op = cv2.imread(img_path)
        img_op = self.predict(img_op)
        out_path = img_path + ".out.jpg"
        cv2.imwrite(out_path, img_op)
        print('[Info] 写入文件完成: {}'.format(out_path))


def main():
    ip = ImgPredicter()
    img_path = os.path.join(IMGS_DIR, 'aoa_yuna.jpg')
    ip.predict_path(img_path)


if __name__ == '__main__':
    main()
