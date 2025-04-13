"""
"""

import cv2
from tensorflow import keras

from lipnet.lipnet_model import LipNet
from lipnet.lipnet_preprocess import VideoFrameData
from util.util import Configuration


class LipnetFeatureExtractor(object):
    def __init__(self, lipnet_config: Configuration):
        self._model_weights = lipnet_config.lipnet_model_weights_path
        self._layer_name = lipnet_config.lipnet_feature_output_layer_name
        self._dlib_model_path = lipnet_config.dlib_model_path
        self._image_width = lipnet_config.image_width
        self._image_height = lipnet_config.image_height
        self._image_channels = lipnet_config.image_channels
        self._max_string = lipnet_config.max_string

    def _get_lipnet_model(self):
        frame_count, image_channels, image_height, image_width, max_string = (
            self._num_frames, self._image_channels, self._image_height,
            self._image_width, self._max_string
        )
        lipnet_model = LipNet(
            frame_count, image_channels, image_height, image_width, max_string
        ).compile_model()
        lipnet_model = lipnet_model.load_weights(self._model_weights)

        layer_output = lipnet_model.model.get_layer(self._layer_name).output
        new_model = keras.Model(inputs=lipnet_model.model.input, outputs=layer_output)

        return new_model

    def _get_num_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return length

    def _process_video(self, video_path):
        video_frame_data = VideoFrameData.get_video_frame_data(
            video_path, self._dlib_model_path
        )
        return self._model.predict(video_frame_data)

    def calculate_predictions(self, video_path):
        self._num_frames = self._get_num_frames(video_path)
        self._model = self._get_lipnet_model()
        return self._process_video(video_path)
