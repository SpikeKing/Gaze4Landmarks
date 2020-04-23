#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/4/17
"""
import time
import copy
import dlib
import os

import cv2
import numpy as np
import tensorflow as tf

from mtcnn.detector import detect_faces
from root_dir import GAZE_MODEL, IMGS_DIR
from utils.mat_utils import center_from_list
from utils.project_utils import traverse_dir_files
from utils.video_utils import show_img_bgr, draw_box


class GazePredicter(object):
    def __init__(self):
        # self.sess = tf.Session()

        # Specify which GPU(s) to use
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0

        # On CPU/GPU placement
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)

        pb_path = os.path.join(GAZE_MODEL, 'good_frozen.pb')
        with tf.gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        self.landmarks_predictor = self.get_landmarks_predictor()
        self.face_detector = self.get_face_detector()
        pass

    def get_model(self, sess):
        frame_index = sess.graph.get_tensor_by_name('Video/fifo_queue_DequeueMany:0')
        eye = sess.graph.get_tensor_by_name('Video/fifo_queue_DequeueMany:1')
        eye_index = sess.graph.get_tensor_by_name('Video/fifo_queue_DequeueMany:2')
        heatmaps = sess.graph.get_tensor_by_name('hourglass/hg_2/after/hmap/conv/BiasAdd:0')
        landmarks = sess.graph.get_tensor_by_name('upscale/mul:0')
        radius = sess.graph.get_tensor_by_name('radius/out/fc/BiasAdd:0')
        sess.run(tf.global_variables_initializer())
        return eye, heatmaps, landmarks, radius

    def eye_preprocess(self, eye):
        _data_format = 'NHWC'

        eye = cv2.equalizeHist(eye)
        eye = eye.astype(np.float32)
        eye *= 2.0 / 255.0
        eye -= 1.0
        eye = np.expand_dims(eye, -1 if _data_format == 'NHWC' else 0)
        return eye

    def detect_eyes(self, face_dict):
        """From found landmarks in previous steps, segment eye image."""
        eyes = []

        # Final output dimensions
        oh, ow = (108, 180)

        landmarks = face_dict['landmarks']

        # Segment eyes
        # for corner1, corner2, is_left in [(36, 39, True), (42, 45, False)]:
        for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
            x1, y1 = landmarks[corner1, :]
            x2, y2 = landmarks[corner2, :]
            eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
            if eye_width == 0.0:
                continue
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

            # Centre image on middle of eye
            translate_mat = np.asmatrix(np.eye(3))
            translate_mat[:2, 2] = [[-cx], [-cy]]
            inv_translate_mat = np.asmatrix(np.eye(3))
            inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

            # Rotate to be upright
            roll = 0.0 if x1 == x2 else np.arctan((y2 - y1) / (x2 - x1))
            rotate_mat = np.asmatrix(np.eye(3))
            cos = np.cos(-roll)
            sin = np.sin(-roll)
            rotate_mat[0, 0] = cos
            rotate_mat[0, 1] = -sin
            rotate_mat[1, 0] = sin
            rotate_mat[1, 1] = cos
            inv_rotate_mat = rotate_mat.T

            # Scale
            scale = ow / eye_width
            scale_mat = np.asmatrix(np.eye(3))
            scale_mat[0, 0] = scale_mat[1, 1] = scale
            inv_scale = 1.0 / scale
            inv_scale_mat = np.asmatrix(np.eye(3))
            inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

            # Centre image
            centre_mat = np.asmatrix(np.eye(3))
            centre_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
            inv_centre_mat = np.asmatrix(np.eye(3))
            inv_centre_mat[:2, 2] = -centre_mat[:2, 2]

            # Get rotated and scaled, and segmented image
            transform_mat = centre_mat * scale_mat * rotate_mat * translate_mat
            inv_transform_mat = (inv_translate_mat * inv_rotate_mat * inv_scale_mat *
                                 inv_centre_mat)
            eye_image = cv2.warpAffine(face_dict['gray'], transform_mat[:2, :], (ow, oh))
            if is_left:
                eye_image = np.fliplr(eye_image)

            eyes.append({
                'image': eye_image,
                'inv_landmarks_transform_mat': inv_transform_mat,
                'side': 'left' if is_left else 'right',
            })

            # print('[Info] eye_image: {}'.format(eye_image.shape))

            # 关键步骤3, 绘制眼睛
            # show_img_grey(eye_image)

        face_dict['eyes'] = eyes

        return face_dict

    def get_faces_v1(self, img_gray):
        """
        人脸检测
        :param face_dict: 数据字典
        :return: 人脸，[x_min, y_min, x_max, y_max]
        """
        detector = self.face_detector
        if detector.__class__.__name__ == 'CascadeClassifier':
            detections = detector.detectMultiScale(img_gray)
        else:
            detections = detector(cv2.resize(img_gray, (0, 0), fx=0.5, fy=0.5), 0)
        faces = []
        for d in detections:
            try:
                l, t, r, b = d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()
                l = max(0, l)
                t = max(0, t)
                l *= 2
                t *= 2
                r *= 2
                b *= 2
            except AttributeError:  # Using OpenCV LBP detector on CPU
                l, t, w, h = d
                r, b = l + w, t + h
            faces.append((l, t, r, b))
            # draw_box(face_dict['img'], l, t, r, b)

        faces.sort(key=lambda bbox: bbox[0])  # 根据左上进行排序
        return faces

    def get_faces_v2(self, face_dict):
        """
        基于MTCNN检测人脸
        :param face_dict: 数据字典
        :return: 人脸检测框
        """
        img_op = face_dict['img']
        bounding_boxes, landmarks = detect_faces(img_op)
        return bounding_boxes

    def get_landmarks_predictor(self):
        """Get a singleton dlib face landmark predictor."""
        dat_path = os.path.join(GAZE_MODEL, 'shape_predictor_5_face_landmarks.dat')
        landmarks_predictor = dlib.shape_predictor(dat_path)
        return landmarks_predictor

    def get_face_detector(self):
        """Get a singleton dlib face detector."""
        try:
            dat_path = os.path.join(GAZE_MODEL, 'mmod_human_face_detector.dat')
            face_detector = dlib.cnn_face_detection_model_v1(dat_path)
        except:
            xml_path = os.path.join(GAZE_MODEL, 'lbpcascade_frontalface_improved.xml')
            face_detector = cv2.CascadeClassifier(xml_path)
        return face_detector

    def detect_landmarks(self, face_dict):
        """Detect 5-point facial landmarks for faces in frame."""
        l, t, w, h = face_dict['box']

        rectangle = dlib.rectangle(left=int(l), top=int(t), right=int(l + w), bottom=int(t + h))
        landmarks_dlib = self.landmarks_predictor(face_dict['gray'], rectangle)

        def tuple_from_dlib_shape(index):
            p = landmarks_dlib.part(index)
            return p.x, p.y

        num_landmarks = landmarks_dlib.num_parts
        landmarks = np.array([tuple_from_dlib_shape(i) for i in range(num_landmarks)])

        # 2. 关键步骤, 绘制关键点
        # draw_points(face_dict['img'], landmarks)
        # print('[Info] landmarks: {}'.format(landmarks.shape))
        face_dict['landmarks'] = landmarks  # 人脸关键点

    def detect_gazes(self, face_dict):
        """
        检测目光关键点
        """
        if len(face_dict['eyes']) != 2:
            return
        eye1 = self.eye_preprocess(face_dict['eyes'][0]['image'])
        eye2 = self.eye_preprocess(face_dict['eyes'][1]['image'])

        eyeI = np.concatenate((eye1, eye2), axis=0)
        eyeI = eyeI.reshape(2, 108, 180, 1)
        # print('[Info] eyeI: {}'.format(eyeI.shape))

        eye, heatmaps, landmarks, radius = self.get_model(self.sess)

        Placeholder_1 = self.sess.graph.get_tensor_by_name('learning_params/Placeholder_1:0')
        feed_dict = {eye: eyeI, Placeholder_1: False}
        oheatmaps, olandmarks, oradius = self.sess.run((heatmaps, landmarks, radius), feed_dict=feed_dict)
        face_dict['gaze'] = (oheatmaps, olandmarks, oradius)

        # print('[Info] oheatmaps: {}, olandmarks: {}, oradius: {}'
        #       .format(oheatmaps.shape, olandmarks.shape, oradius))

        face_dict['oheatmaps'] = oheatmaps  # 热力图
        face_dict['olandmarks'] = olandmarks  # 关键点
        face_dict['oradius'] = oradius  # 眼睛半径

    def draw_gaze_eye(self, face_dict):
        face_dict = copy.deepcopy(face_dict)
        print('[Info] 绘制!')
        th = 3
        for i in range(2):
            eye_landmarks = face_dict['olandmarks'][i]
            eye_image = face_dict['eyes'][i]['image']
            eye_radius = face_dict['oradius'][i][0]
            eye_side = face_dict['eyes'][i]['side']

            if eye_side == 'left':
                eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
                eye_image = np.fliplr(eye_image)

            eye_upscale = 2
            eye_image = cv2.equalizeHist(eye_image)
            eye_image_raw = cv2.cvtColor(eye_image, cv2.COLOR_GRAY2BGR)
            eye_image_raw = cv2.resize(eye_image_raw, (0, 0), fx=eye_upscale, fy=eye_upscale)

            # 眼睑
            eye_image_annotated = np.copy(eye_image_raw)
            # print('[Info] 眼睑: {}'.format(eye_landmarks[0:8]))
            cv2.polylines(
                eye_image_annotated,
                [np.round(eye_upscale * eye_landmarks[0:8]).astype(np.int32)
                     .reshape(-1, 1, 2)],
                isClosed=True, color=(255, 255, 0),
                thickness=th, lineType=cv2.LINE_AA,
            )
            # show_img_bgr(eye_image_annotated)  # 眼睑图像

            # 虹膜
            cv2.polylines(
                eye_image_annotated,
                [np.round(eye_upscale * eye_landmarks[8:16]).astype(np.int32)
                     .reshape(-1, 1, 2)],
                isClosed=True, color=(0, 255, 255),
                thickness=th, lineType=cv2.LINE_AA,
            )
            # show_img_bgr(eye_image_annotated)  # 虹膜图像

            iris_center = eye_landmarks[16]
            eyeball_center = np.array(center_from_list(eye_landmarks[0:8]))

            # 虹膜中心
            cv2.drawMarker(
                eye_image_annotated,
                tuple(np.round(eye_upscale * iris_center).astype(np.int32)),
                color=(255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=4,
                thickness=th + 1, line_type=cv2.LINE_AA,
            )
            # show_img_bgr(eye_image_annotated)  # 瞳孔中心
            # import os
            # from root_dir import IMGS_DIR
            # img_path = os.path.join(IMGS_DIR, 'xxx.iris_center.{}.jpg'.format(str(eye_side)))
            # cv2.imwrite(img_path, eye_image_annotated)

            # 眼睑中心
            cv2.drawMarker(
                eye_image_annotated,
                tuple(np.round(eye_upscale * eyeball_center).astype(np.int32)),
                color=(255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=4,
                thickness=th + 1, line_type=cv2.LINE_AA,
            )
            # show_img_bgr(eye_image_annotated)  # 瞳孔中心
            # import os
            # from root_dir import IMGS_DIR
            # img_path = os.path.join(IMGS_DIR, 'xxx.eyeball_center.{}.jpg'.format(str(eye_side)))
            # cv2.imwrite(img_path, eye_image_annotated)

            eyeball_radius = np.linalg.norm(iris_center - eyeball_center)
            # print('[Info] eyeball_radius: {}'.format(eyeball_radius))

            i_x0, i_y0 = iris_center
            e_x0, e_y0 = eyeball_center

            theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
            phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)), -1.0, 1.0))

            current_gaze = np.array([theta, phi])
            image_out = self.draw_arrow(
                eye_image_annotated,
                tuple(np.round(eye_upscale * eye_landmarks[16]).astype(np.int32)),
                current_gaze,
                length=25.0 * eye_upscale,
                thickness=th + 2
            )
            show_img_bgr(image_out)  # 瞳孔中心
            # import os
            # from root_dir import IMGS_DIR
            # img_path = os.path.join(IMGS_DIR, 'xxx.gaze.{}.jpg'.format(str(eye_side)))
            # cv2.imwrite(img_path, image_out)

    def get_eyes_closed(self, two_eye_landmarks):
        """
        检测是否闭眼
        """
        eye_openings = []
        for i in range(2):
            eye_landmarks = two_eye_landmarks[i]
            v_low_1 = abs(eye_landmarks[2][1] - eye_landmarks[6][1])  # 中间位置高度
            v_low_2 = abs(eye_landmarks[1][1] - eye_landmarks[7][1])  # 中间位置高度
            v_low_3 = abs(eye_landmarks[3][1] - eye_landmarks[5][1])  # 中间位置高度
            v_low = (v_low_1 + v_low_2 + v_low_3) / 3
            v_wide = abs(eye_landmarks[4][0] - eye_landmarks[0][0])
            ratio = v_low / v_wide
            # print('[Info] v_low: {}, v_wide: {}, ratio: {}'.format(v_low, v_wide, ratio))
            if ratio > 0.12:
                eye_openings.append(True)
                print('[Info] {} 睁眼!'.format(i))
            else:
                eye_openings.append(False)
                print('[Info] {} 闭眼!'.format(i))
        return eye_openings

    def draw_gaze_img(self, face_dict):
        face_dict = copy.deepcopy(face_dict)

        th = 2
        for i in range(2):
            img_op = face_dict['img']
            eye_landmarks = face_dict['olandmarks'][i]
            eye_image = face_dict['eyes'][i]['image']
            eye_radius = face_dict['oradius'][i][0]
            eye = face_dict['eyes'][i]
            eye_side = face_dict['eyes'][i]['side']

            if eye_side == 'left':
                eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
                eye_image = np.fliplr(eye_image)

            eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)), 'constant', constant_values=1.0))
            eye_landmarks = (eye_landmarks * eye['inv_landmarks_transform_mat'].T)[:, :2]
            eye_landmarks = np.asarray(eye_landmarks)

            # 眼睛范围
            cv2.polylines(
                img_op,
                [np.round(eye_landmarks[0:8]).astype(np.int32).reshape(-1, 1, 2)],
                isClosed=True, color=(255, 255, 0),
                thickness=th, lineType=cv2.LINE_AA,
            )
            # show_img_bgr(img_op)

            # 虹膜范围
            cv2.polylines(  # 虹膜
                img_op,
                [np.round(eye_landmarks[8:16]).astype(np.int32).reshape(-1, 1, 2)],
                isClosed=True, color=(0, 255, 255),
                thickness=th, lineType=cv2.LINE_AA,
            )
            # show_img_bgr(img_op)

            # 计算视线方向强度
            iris_center = eye_landmarks[16]
            eyeball_center = eye_landmarks[17]
            eyeball_radius = np.linalg.norm(eyeball_center - iris_center) / eye_radius * 100

            # 绘制边界
            # cv2.drawMarker(
            #     img_op,
            #     tuple(np.round(eyeball_margin).astype(np.int32)),
            #     color=(0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=4,
            #     thickness=th + 3, line_type=cv2.LINE_AA,
            # )
            # show_img_bgr(img_op)  # 瞳孔中心

            # 绘制瞳孔中心
            cv2.drawMarker(
                img_op,
                tuple(np.round(iris_center).astype(np.int32)),
                color=(255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=4,
                thickness=th + 3, line_type=cv2.LINE_AA,
            )
            # show_img_bgr(img_op)  # 瞳孔中心

            # 绘制眼睛中心
            cv2.drawMarker(
                img_op,
                tuple(np.round(eyeball_center).astype(np.int32)),
                color=(255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=4,
                thickness=th + 3, line_type=cv2.LINE_AA,
            )
            # show_img_bgr(img_op)  # 眼睛中心

            i_x0, i_y0 = iris_center
            e_x0, e_y0 = eyeball_center
            theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
            phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)), -1.0, 1.0))
            current_gaze = np.array([theta, phi])

            length = int(eyeball_radius * 2)
            dx = -length * np.sin(current_gaze[1])
            dy = -length * np.sin(current_gaze[0])
            start_p = np.round(iris_center)
            end_p = np.round([iris_center[0] + dx, iris_center[1] + dy])
            v_bias_1 = np.linalg.norm(start_p - end_p)

            v_low_1 = abs(eye_landmarks[2][1] - eye_landmarks[6][1])
            v_low_2 = abs(eye_landmarks[1][1] - eye_landmarks[7][1])
            v_low_3 = abs(eye_landmarks[3][1] - eye_landmarks[5][1])
            v_low = (v_low_1 + v_low_2 + v_low_3) / 3  # 中间位置均值

            intensity = v_bias_1 / v_low
            # print('[Info] 强度: {}'.format(intensity))
            if intensity < 1:
                print('[Info] 眼睛正视 {}!'.format(intensity))
            else:
                print('[Info] 眼睛斜视 {}!'.format(intensity))
                img_op = self.draw_arrow(
                    img_op,
                    tuple(np.round(iris_center).astype(np.int32)),
                    current_gaze,
                    length=int(eyeball_radius * 2),
                    thickness=th
                )

            face_dict['img_draw'] = img_op
            # show_img_bgr(img_op)  # 瞳孔中心

        return face_dict

    @staticmethod
    def draw_arrow(image_in, eye_pos, pitchyaw, length=40.0, thickness=2, color=(0, 0, 255)):
        """Draw gaze angle on given image with a given eye positions."""
        image_out = image_in
        if len(image_out.shape) == 2 or image_out.shape[2] == 1:
            image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
        dx = -length * np.sin(pitchyaw[1])
        dy = -length * np.sin(pitchyaw[0])

        v_bias = np.linalg.norm(np.round(eye_pos) - np.round([eye_pos[0] + dx, eye_pos[1] + dy]))
        # print('[Info] 强度: {}'.format(v_bias))

        cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                        tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                        thickness, cv2.LINE_AA, tipLength=0.2)
        return image_out

    def predict_path(self, img_path):
        """
        预测图像路径
        :param img_path: 视频路径
        :return: 数据字典
        """
        print('\n[Info] 图像路径: {}'.format(img_path))
        img_op = cv2.imread(img_path)  # 读取图像
        face_dict = self.predict(img_op)
        return face_dict

    def predict(self, img_bgr):
        """
        预测图像numpy, 
        :param img_bgr: OpenCV格式图像，BGR通道
        :return: 数据字典
        """
        print('[Info] 预测开始!')
        s_time = time.time()
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # 转换灰度图像
        print('[Info] 图像尺寸: {}'.format(img_bgr.shape))

        face_dict = dict()

        face_dict['img'] = img_bgr  # 图像
        face_dict['gray'] = img_gray  # 灰度图像

        bounding_boxes = self.get_faces_v1(img_gray)
        e_time_1 = time.time()
        print('[Info] 人脸数量: {} / {}'.format(len(bounding_boxes), e_time_1 - s_time))

        for box in bounding_boxes:  # 遍历人脸数量
            s_time_1 = time.time()
            x_min, y_min, x_max, y_max = box

            # 1. 关键步骤: 绘制人脸
            # draw_box(img_op, x_min, y_min, x_max, y_max)  # 绘制人脸

            x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
            face_dict['box'] = x, y, w, h
            self.detect_landmarks(face_dict)
            e_time_2 = time.time()
            print('[Info] 人脸Landmarks耗时: {}'.format(e_time_2 - s_time_1))

            self.detect_eyes(face_dict)  # 检测眼睛
            e_time_3 = time.time()
            print('[Info] 眼睛矫正耗时: {}'.format(e_time_3 - e_time_2))

            self.detect_gazes(face_dict)
            e_time_4 = time.time()
            print('[Info] 眼睛Landmarks耗时: {}'.format(e_time_4 - e_time_3))

            two_eye_landmarks = face_dict['olandmarks']
            two_eye_openings = self.get_eyes_closed(two_eye_landmarks)
            # self.draw_gaze_eye(face_dict)  # 绘制效果

            face_dict = self.draw_gaze_img(face_dict)

        # img_op = face_dict['img_draw']
        # show_img_bgr(img_op)  # 瞳孔中心

        return face_dict


def folder_test():
    img_dir = os.path.join(IMGS_DIR, 'tests')
    out_dir = os.path.join(IMGS_DIR, 'tests-out')

    paths_list, names_list = traverse_dir_files(img_dir)
    gp = GazePredicter()

    for img_path, name in zip(paths_list, names_list):
        print('[Info] 处理图像: {}'.format(name))
        face_dict = gp.predict_path(img_path)
        img_op = face_dict['img_draw']
        out_path = os.path.join(out_dir, name + ".out.jpg")
        cv2.imwrite(out_path, img_op)


def main():
    img_path = os.path.join(IMGS_DIR, 'kkk-x.jpg')

    gp = GazePredicter()
    face_dict = gp.predict_path(img_path)
    img_op = face_dict['img_draw']
    out_path = img_path + ".out.jpg"
    cv2.imwrite(out_path, img_op)


if __name__ == '__main__':
    # main()
    folder_test()
