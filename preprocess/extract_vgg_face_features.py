import argparse
import logging
import os
import pickle

import numpy
from keras.engine import Model
from keras_vggface import utils
from keras_vggface.vggface import VGGFace

from util.logging_setup import setup_logging
from util.util import Configuration, get_files_recursively_ext

_logger = logging.getLogger(__name__)


def load_all_data(data_folder: str) -> dict:
    merged_data = dict()
    all_pickle_files_in_folder = get_files_recursively_ext(data_folder, ["pkl"])
    for pkl_file_path in all_pickle_files_in_folder:
        with open(pkl_file_path, "rb") as istream:
            merged_data.update(
                pickle.load(istream)
            )

    return merged_data


def preprocess_face_crop(cv2_face_crop):
    # this also converts it to floating point precision
    channel_reversed_img = cv2_face_crop[..., :: -1].astype(numpy.float32)
    expanded_img = numpy.expand_dims(channel_reversed_img, axis=0)
    preprocessed_img = utils.preprocess_input(expanded_img, version=1)  # or version=2

    return preprocessed_img


def get_vgg_layer_model(model_name: str, layer_name: str):
    vgg_model = VGGFace(model=model_name)
    print("Printing model summary:")
    # print the model_summary
    vgg_model.summary()

    out = vgg_model.get_layer(layer_name).output
    vgg_model_new = Model(vgg_model.input, out)

    return vgg_model_new


def preprocess_face_list(face_list, model):
    face_features_list = list()
    for face in face_list:
        preprocessed_face = preprocess_face_crop(face)
        cnn_features = model.predict(preprocessed_face)
        cnn_features = numpy.squeeze(cnn_features)
        face_features_list.append(cnn_features)

    return face_features_list


def main(vgg_face_config: Configuration) -> None:
    _logger.info("loading data")
    extracted_faces_dict = load_all_data(vgg_face_config.extracted_faces_path)

    _logger.info("loading model")
    model = get_vgg_layer_model(
        model_name=vgg_face_config.model_name,
        layer_name=vgg_face_config.layer_name
    )

    counter = 1
    total_size = len(extracted_faces_dict)
    output_dict = dict()
    for video_name, extracted_faces_list in extracted_faces_dict.items():
        _logger.info(f"processing {counter} out of {total_size} -> {counter / total_size * 100:.2f}%")

        output_dict[video_name] = preprocess_face_list(
            face_list=extracted_faces_list,
            model=model
        )
        counter += 1

    _logger.info(f"saving to {vgg_face_config.save_path}")
    with open(vgg_face_config.save_path, "wb") as ostream:
        pickle.dump(output_dict, ostream)

    _logger.info("done")


if __name__ == '__main__':
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
        os.path.join(args.config_dir, "vgg_face_config.yml")
    )

    main(vgg_face_config=vgg_config)
