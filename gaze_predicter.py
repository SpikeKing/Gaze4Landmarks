#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2020/4/17
"""
import copy
import dlib
import os

import cv2
import numpy as np
import tensorflow as tf

from mtcnn.detector import detect_faces
from root_dir import GAZE_MODEL, IMGS_DIR
from utils.mat_utils import center_from_list
from utils.video_utils import show_img_bgr, draw_points


class GazePredicter(object):
    def __init__(self):
        self.sess = tf.Session()
        self.pb_path = os.path.join(GAZE_MODEL, 'good_frozen.pb')
        pass

    def get_model(self, sess, pb_path):
        with tf.gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

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

            print('[Info] eye_image: {}'.format(eye_image.shape))

            # 关键步骤3, 绘制眼睛
            # show_img_grey(eye_image)

        face_dict['eyes'] = eyes

        return face_dict

    def get_landmarks_predictor(self):
        """Get a singleton dlib face landmark predictor."""
        dat_path = os.path.join(GAZE_MODEL, 'shape_predictor_5_face_landmarks.dat')
        landmarks_predictor = dlib.shape_predictor(dat_path)
        return landmarks_predictor

    def detect_landmarks(self, face_dict):
        """Detect 5-point facial landmarks for faces in frame."""
        predictor = self.get_landmarks_predictor()
        l, t, w, h = face_dict['box']
        rectangle = dlib.rectangle(left=int(l), top=int(t), right=int(l + w), bottom=int(t + h))
        landmarks_dlib = predictor(face_dict['gray'], rectangle)

        def tuple_from_dlib_shape(index):
            p = landmarks_dlib.part(index)
            return p.x, p.y

        num_landmarks = landmarks_dlib.num_parts
        landmarks = np.array([tuple_from_dlib_shape(i) for i in range(num_landmarks)])

        # 2. 关键步骤, 绘制关键点
        draw_points(face_dict['img'], landmarks)
        print('[Info] landmarks: {}'.format(landmarks.shape))

        face_dict['landmarks'] = landmarks

    def detect_gazes(self, face_dict):
        if len(face_dict['eyes']) != 2:
            return
        eye1 = self.eye_preprocess(face_dict['eyes'][0]['image'])
        eye2 = self.eye_preprocess(face_dict['eyes'][1]['image'])

        eyeI = np.concatenate((eye1, eye2), axis=0)
        eyeI = eyeI.reshape(2, 108, 180, 1)
        print('[Info] eyeI: {}'.format(eyeI.shape))

        eye, heatmaps, landmarks, radius = self.get_model(self.sess, self.pb_path)

        Placeholder_1 = self.sess.graph.get_tensor_by_name('learning_params/Placeholder_1:0')
        feed_dict = {eye: eyeI, Placeholder_1: False}
        oheatmaps, olandmarks, oradius = self.sess.run((heatmaps, landmarks, radius), feed_dict=feed_dict)
        face_dict['gaze'] = (oheatmaps, olandmarks, oradius)
        print('[Info] oheatmaps: {}, olandmarks: {}, oradius: {}'
              .format(oheatmaps.shape, olandmarks.shape, oradius))

        face_dict['oheatmaps'] = oheatmaps
        face_dict['olandmarks'] = olandmarks
        face_dict['oradius'] = oradius

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
            # import os
            # from root_dir import IMGS_DIR
            # img_path = os.path.join(IMGS_DIR, 'xxx.eyeball.{}.jpg'.format(str(eye_side)))
            # cv2.imwrite(img_path, eye_image_annotated)

            # 虹膜
            cv2.polylines(
                eye_image_annotated,
                [np.round(eye_upscale * eye_landmarks[8:16]).astype(np.int32)
                     .reshape(-1, 1, 2)],
                isClosed=True, color=(0, 255, 255),
                thickness=th, lineType=cv2.LINE_AA,
            )
            # show_img_bgr(eye_image_annotated)  # 虹膜图像
            # import os
            # from root_dir import IMGS_DIR
            # img_path = os.path.join(IMGS_DIR, 'xxx.iris.{}.jpg'.format(str(eye_side)))
            # cv2.imwrite(img_path, eye_image_annotated)

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

            iris_center = eye_landmarks[16]
            # eyeball_center = np.array(center_from_list(eye_landmarks[0:8]))
            eyeball_center = eye_landmarks[17]
            eyeball_margin = [eyeball_center[0] + eye_radius, eyeball_center[1]]
            eyeball_radius = np.linalg.norm(eyeball_margin - eyeball_center)
            # eyeball_radius = np.linalg.norm(iris_center - eyeball_center)
            # print('[Info] eyeball_radius: {}'.format(eyeball_radius))

            cv2.drawMarker(
                img_op,
                tuple(np.round(iris_center).astype(np.int32)),
                color=(255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=4,
                thickness=th + 3, line_type=cv2.LINE_AA,
            )
            # show_img_bgr(eye_image_annotated)  # 瞳孔中心

            cv2.drawMarker(
                img_op,
                tuple(np.round(eyeball_center).astype(np.int32)),
                color=(255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=4,
                thickness=th + 3, line_type=cv2.LINE_AA,
            )
            # show_img_bgr(eye_image_annotated)  # 眼睛中心

            i_x0, i_y0 = iris_center
            e_x0, e_y0 = eyeball_center
            theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
            phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)), -1.0, 1.0))
            current_gaze = np.array([theta, phi])
            img_op = self.draw_arrow(
                img_op,
                tuple(np.round(iris_center).astype(np.int32)),
                current_gaze,
                length=int(eyeball_radius * 2),
                thickness=th
            )

            face_dict['img_draw'] = img_op
            # show_img_bgr(img_op)  # 瞳孔中心

        # import os
        # from root_dir import IMGS_DIR
        # img_path = os.path.join(IMGS_DIR, 'xxx.final.{}.jpg'.format(str(eye_side)))
        # cv2.imwrite(img_path, image_out)
        return face_dict

    @staticmethod
    def draw_arrow(image_in, eye_pos, pitchyaw, length=40.0, thickness=2, color=(0, 0, 255)):
        """Draw gaze angle on given image with a given eye positions."""
        image_out = image_in
        if len(image_out.shape) == 2 or image_out.shape[2] == 1:
            image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
        dx = -length * np.sin(pitchyaw[1])
        dy = -length * np.sin(pitchyaw[0])
        cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                        tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                        thickness, cv2.LINE_AA, tipLength=0.2)
        return image_out

    def predict_path(self, img_path):
        # img_path = os.path.join(IMGS_DIR, 'aoa_yuna.jpg')
        img_op = cv2.imread(img_path)
        face_dict = self.predict(img_op)
        return face_dict

    def predict(self, img_op):
        print('[Info] 预测人脸!')
        img_gray = cv2.cvtColor(img_op, cv2.COLOR_BGR2GRAY)
        print('[Info] 图像尺寸: {}'.format(img_op.shape))
        bounding_boxes, landmarks = detect_faces(img_op)
        print('[Info] 人脸检测: {}'.format(landmarks))
        face_dict = dict()
        face_dict['img'] = img_op
        face_dict['gray'] = img_gray

        for box in bounding_boxes:
            x_min, y_min, x_max, y_max, _ = box

            # 1. 关键步骤: 绘制人脸
            # draw_box(img_op, x_min, y_min, x_max, y_max)  # 绘制人脸

            x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min
            face_dict['box'] = x, y, w, h

            self.detect_landmarks(face_dict)
            self.detect_eyes(face_dict)
            self.detect_gazes(face_dict)
            # self.draw_gaze_eye(face_dict)
            face_dict = self.draw_gaze_img(face_dict)

        # img_op = face_dict['img_draw']
        # show_img_bgr(img_op)  # 瞳孔中心

        return face_dict


def main():
    # img_path = os.path.join(IMGS_DIR, 'xxx.jpg')

    # img_path = os.path.join(IMGS_DIR, 'kkk-1.jpg')
    # img_path = os.path.join(IMGS_DIR, 'kkk-2.jpg')
    # img_path = os.path.join(IMGS_DIR, 'kkk-5.jpg')
    # img_path = os.path.join(IMGS_DIR, 'kkk-6.jpg')

    # for i in range(1, 8):

    img_path = os.path.join(IMGS_DIR, 'kkk-{}g.jpg'.format(7))

    gp = GazePredicter()
    face_dict = gp.predict_path(img_path)
    img_op = face_dict['img_draw']
    out_path = img_path + ".out.jpg"
    cv2.imwrite(out_path, img_op)


if __name__ == '__main__':
    main()
