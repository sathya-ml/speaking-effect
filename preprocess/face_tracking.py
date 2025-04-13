# mtcnn detector
import cv2
import numpy as np
from mtcnn import MTCNN


class VideoFaceDetectorKalman(object):
    def __init__(self, downsample_rate):
        self._detector = MTCNN()
        self._t_downsample_rate = downsample_rate

    def detect_faces_in_video(self, video_path):
        video_capture = cv2.VideoCapture(video_path)
        success, frame = video_capture.read()
        frame_count = 0
        kalman_is_init = False
        faces = list()
        while success:
            if not kalman_is_init:
                detection = self._detector.detect_faces(frame)
                if len(detection) == 0:
                    frame_count += 1
                    success, frame = video_capture.read()
                    continue

                [x0, y0, w0, h0] = detection[0]['box']
                w0 = 2 * (int(w0 / 2))
                h0 = 2 * (int(h0 / 2))

                # Cropping face
                crop = frame[y0:y0 + h0, x0:x0 + w0, :]

                self.cropSize = crop.shape[:2]
                faces.append(crop)

                # set the initial tracking window
                state = np.array([int(x0 + w0 / 2), int(y0 + h0 / 2), 0, 0], dtype='float64')  # initial position

                # Setting up Kalman Filter
                self._kalman = cv2.KalmanFilter(4, 2, 0)
                self._kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                                          [0., 1., 0., .1],
                                                          [0., 0., 1., 0.],
                                                          [0., 0., 0., 1.]])
                self._kalman.measurementMatrix = 1. * np.eye(2, 4)
                self._kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
                self._kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
                self._kalman.errorCovPost = 1e-1 * np.eye(4, 4)
                self._kalman.statePost = state
                # measurement = np.array([int(x0 + w0 / 2), int(y0 + h0 / 2)], dtype='float64')
                kalman_is_init = True
            else:
                if frame_count % self._t_downsample_rate == 0:
                    detection = self._detector.detect_faces(frame)
                    if len(detection) != 0:
                        [x0, y0, w, h] = detection[0]['box']
                        not_found = False
                    else:
                        not_found = True
                    if len(detection) > 1:
                        print(f"error: found {len(detection)} faces in frame {frame_count}")

                prediction = self._kalman.predict()  # prediction

                if frame_count % self._t_downsample_rate == 0 and not not_found:
                    measurement = np.array([x0 + w / 2, y0 + h / 2], dtype='float64')
                    posterior = self._kalman.correct(measurement)
                    [cx0, cy0, wn, hn] = posterior.astype(int)
                else:
                    [cx0, cy0, wn, hn] = prediction.astype(int)

                # Cropping with new bounding box
                crop = frame[int(cy0 - h0 / 2):int(cy0 + h0 / 2), int(cx0 - w0 / 2):int(cx0 + w0 / 2), :]

                faces.append(crop.astype('uint8').copy())

                frame_count += 1
                success, frame = video_capture.read()
        return faces


def __test(path):
    vfd = VideoFaceDetectorKalman(downsample_rate=10)
    faces = vfd.detect_faces_in_video(path)
    for i, f in enumerate(faces):
        cv2.imwrite(f"im_{i}.jpg", f)


if __name__ == '__main__':
    VIDEO_PATH = ...  # Input video path for a quick test
    __test(VIDEO_PATH)

