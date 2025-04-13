import argparse
import logging
import os
import pickle

import cv2
from PIL import Image

from util.logging_setup import setup_logging
from util.util import Configuration, get_files_recursively_ext
from vgg19_fer_net.emotion_pred_net import FacialEmotionClassifier

_logger = logging.getLogger(__name__)


class VideoEmotionFeatureExtractor(object):
    def __init__(self, emotion_predictor: FacialEmotionClassifier):
        self._logger = logging.getLogger(VideoEmotionFeatureExtractor.__name__)

        self._emotion_predictor = emotion_predictor

    def preprocess_frame(self, cv2_img):
        img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    def extract_video_emotion_features(self, video_name, cropped_faces):
        self._logger.debug(f"found {len(cropped_faces)} cropped faces for {video_name}")

        features_list = list()
        for cropped_face in cropped_faces:
            preprocessed_face = self.preprocess_frame(cropped_face)
            face_features = self._emotion_predictor.eval_img_from_cropped_face(preprocessed_face)

            features_list.append(
                face_features.cpu().numpy()
            )
        return features_list


def load_all_data(data_folder: str) -> dict:
    merged_data = dict()
    all_pickle_files_in_folder = get_files_recursively_ext(data_folder, ["pkl"])
    for pkl_file_path in all_pickle_files_in_folder:
        with open(pkl_file_path, "rb") as istream:
            merged_data.update(
                pickle.load(istream)
            )

    return merged_data


def main(vgg_emotion_config: Configuration) -> None:
    _logger.info("loading data")
    extracted_faces_dict = load_all_data(vgg_emotion_config.extracted_faces_path)

    _logger.info("loading model")
    facial_emotion_classifier = FacialEmotionClassifier.get_facial_emotion_classifier(
        model_name=vgg_emotion_config.model_name, model_path=vgg_emotion_config.model_path
    )
    feature_extractor = VideoEmotionFeatureExtractor(
        emotion_predictor=facial_emotion_classifier
    )

    counter = 1
    total_size = len(extracted_faces_dict)
    output_dict = dict()
    for video_name, extracted_faces_list in extracted_faces_dict.items():
        _logger.info(f"processing {counter} out of {total_size} -> {counter / total_size * 100:.2f}%")

        output_dict[video_name] = feature_extractor.extract_video_emotion_features(
            video_name=video_name, cropped_faces=extracted_faces_list
        )
        counter += 1

    _logger.info(f"saving to {vgg_emotion_config.save_path}")
    with open(vgg_emotion_config.save_path, "wb") as ostream:
        pickle.dump(output_dict, ostream)

    _logger.info("done")


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-c", "--config_dir", required=True,
        help="path to the directory containing config files"
    )
    args = arg_parser.parse_args()

    setup_logging(
        os.path.join(args.config_dir, "logging_config.yml")
    )

    vgg_config = Configuration.load_from_yaml_file(
        os.path.join(args.config_dir, "vgg_emotion_config.yml")
    )

    main(vgg_emotion_config=vgg_config)
