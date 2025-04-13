import logging
from typing import Optional

import cv2
import dlib
import numpy
import skvideo.io
from imutils import face_utils
from tensorflow import keras

import lipnet.lipnet_env as env

FRAME_SHAPE = (env.IMAGE_HEIGHT, env.IMAGE_WIDTH, env.IMAGE_CHANNELS)
IMAGE_SIZE = (env.IMAGE_HEIGHT, env.IMAGE_WIDTH)

_logger = logging.getLogger(__name__)


class ImgUtil(object):
    @staticmethod
    def crop_image(image: numpy.ndarray, center: tuple, size: tuple) -> numpy.ndarray:
        start = tuple(a - b // 2 for a, b in zip(center, size))
        end = tuple(a + b for a, b in zip(start, size))
        slices = tuple(slice(a, b) for a, b in zip(start, end))

        return image[slices]

    @staticmethod
    def swap_center_axis(t: numpy.ndarray) -> tuple:
        return t[1], t[0]


class ReshapeAndNormalize(object):
    @staticmethod
    def reshape_video_data(video_data: numpy.ndarray) -> numpy.ndarray:
        reshaped_video_data = numpy.swapaxes(video_data, 1, 2)  # T x W x H x C

        if keras.backend.image_data_format() == 'channels_first':
            reshaped_video_data = numpy.rollaxis(reshaped_video_data, 3)  # C x T x W x H

        return reshaped_video_data

    @staticmethod
    def normalize_video_data(video_data: numpy.ndarray) -> numpy.ndarray:
        return video_data.astype(numpy.float32) / 255

    @staticmethod
    def reshape_and_normalize_video_data(video_data: numpy.ndarray) -> numpy.ndarray:
        return ReshapeAndNormalize.normalize_video_data(
            ReshapeAndNormalize.reshape_video_data(video_data)
        )


class ExtractFeatures(object):
    @staticmethod
    def get_mouth_points_center(mouth_points: numpy.ndarray) -> numpy.ndarray:
        mouth_centroid = numpy.mean(mouth_points[:, -2:], axis=0, dtype=int)
        return mouth_centroid

    @staticmethod
    def extract_mouth_points(frame: numpy.ndarray, detector, predictor) -> Optional[numpy.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = detector(gray, 1)

        if len(detected) <= 0:
            return None

        shape = face_utils.shape_to_np(predictor(gray, detected[0]))
        _, (i, j) = list(face_utils.FACIAL_LANDMARKS_IDXS.items())[0]

        return numpy.array([shape[i:j]][0])

    @staticmethod
    def extract_mouth_on_frame(frame: numpy.ndarray, detector, predictor, idx: int) -> Optional[numpy.ndarray]:
        m_points = ExtractFeatures.extract_mouth_points(frame, detector, predictor)

        if m_points is None:
            _logger.warning(f"No ROI found at frame {idx}")
            return None

        m_center = ExtractFeatures.get_mouth_points_center(m_points)
        s_m_center = ImgUtil.swap_center_axis(m_center)

        crop = ImgUtil.crop_image(frame, s_m_center, IMAGE_SIZE)

        if crop.shape != FRAME_SHAPE:
            _logger.warning("Wrong shape {crop.shape} at frame {idx}")
            return None

        return crop

    @staticmethod
    def extract_video_data(path: str, detector, predictor) -> list:
        video_data = skvideo.io.vread(path)
        mouth_data = []

        for i, f in enumerate(video_data):
            c = ExtractFeatures.extract_mouth_on_frame(f, detector, predictor, i)
            mouth_data.append(c)

        return mouth_data


class VideoFrameData(object):
    @staticmethod
    def _replace_none(lst):
        # Initialize with None to indicate no valid frame seen yet
        last_valid = None
        
        for item in lst:
            if item is None:
                if last_valid is None:
                    # If we haven't seen any valid frames yet, skip this frame
                    continue
                else:
                    # Use the last valid frame we've seen
                    yield last_valid
            else:
                # Update the last valid frame and yield it
                last_valid = item
                yield item

    @staticmethod
    def get_video_frame_data(video_path: str, predictor_path) -> [numpy.ndarray]:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)

        data = ExtractFeatures.extract_video_data(video_path, detector, predictor)

        none_count = sum(list(item for item in data if data is None))
        if none_count > 0:
            _logger.warning(f"found {none_count} None elements in {video_path}")

        clean_data = numpy.array(list(VideoFrameData._replace_none(data)))
        normalized_data = ReshapeAndNormalize.reshape_and_normalize_video_data(clean_data)
        return numpy.array([normalized_data])
