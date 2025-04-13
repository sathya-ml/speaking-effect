import argparse
import logging
import os
import pickle

import numpy
import pandas
from scipy import stats
import sklearn.metrics as metrics
from tensorflow import keras

from preprocess import load_dataset_features
from util.logging_setup import setup_logging
from util.util import Configuration, find_files_in_folder_by_format

_BATCH_SIZE = 32
_FACE_MODALITY = "emotion"


def get_model_weights_and_stats_paths(model_dir, num_frames):
    all_files = find_files_in_folder_by_format(model_dir, ["npy", "csv"])

    cnn_models = [file for file in all_files if "cnn" in file]
    rnn_models = [file for file in all_files if "gru" in file]

    cnn_lip_models = [file for file in cnn_models if "lip" in file]
    cnn_pure_models = [file for file in cnn_models if file not in cnn_lip_models]

    rnn_lip_models = [file for file in rnn_models if "lip" in file]
    rnn_pure_models = [file for file in rnn_models if file not in rnn_lip_models]

    cnn = [
        file for file in cnn_pure_models
        if f"{num_frames:02}" in file and ".npy" in file
    ]
    cnn_log = [
        file for file in cnn_pure_models
        if f"{num_frames:02}" in file and ".csv" in file
    ]

    cnn_lip = [
        file for file in cnn_lip_models
        if f"{num_frames:02}" in file and ".npy" in file
    ]
    cnn_lip_log = [
        file for file in cnn_lip_models
        if f"{num_frames:02}" in file and ".csv" in file
    ]

    rnn = [
        file for file in rnn_pure_models
        if f"{num_frames:02}" in file and ".npy" in file
    ]
    rnn_log = [
        file for file in rnn_pure_models
        if f"{num_frames:02}" in file and ".csv" in file
    ]

    rnn_lip = [
        file for file in rnn_lip_models
        if f"{num_frames:02}" in file and ".npy" in file
    ]
    rnn_lip_log = [
        file for file in rnn_lip_models
        if f"{num_frames:02}" in file and ".csv" in file
    ]

    return {
        "cnn": (cnn[0], cnn_log[0]),
        "cnn_lip": (cnn_lip[0], cnn_lip_log[0]),
        "rnn": (rnn[0], rnn_log[0]),
        "rnn_lip": (rnn_lip[0], rnn_lip_log[0])
    }


def get_frame_model_weights_and_stats_path(model_dir):
    all_files = find_files_in_folder_by_format(model_dir, ["npy", "csv"])
    frame_model = [file for file in all_files if "frame" in file]

    return (
        [file for file in frame_model if ".npy" in file][0],
        [file for file in frame_model if ".csv" in file][0]
    )


def construct_test_data_generators(data_dir, test_actors, frame_num):
    cnn_test_data_generator = load_dataset_features.DataGeneratorFaceCNN(
        data_folder=data_dir,
        test_actors=test_actors,
        frame_num=frame_num,
        batch_size=_BATCH_SIZE,
        face_modality=_FACE_MODALITY,
        is_training=False
    )

    cnn_lip_test_data_generator = load_dataset_features.DataGeneratorLipCNN(
        data_folder=data_dir,
        test_actors=test_actors,
        frame_num=frame_num,
        batch_size=_BATCH_SIZE,
        face_modality=_FACE_MODALITY,
        is_training=False
    )

    rnn_test_data_generator = load_dataset_features.DataGeneratorFaceRNN(
        data_folder=data_dir,
        test_actors=test_actors,
        frame_num=frame_num,
        batch_size=_BATCH_SIZE,
        face_modality=_FACE_MODALITY,
        is_training=False
    )

    rnn_lip_test_data_generator = load_dataset_features.DataGeneratorLipRNN(
        data_folder=data_dir,
        test_actors=test_actors,
        frame_num=frame_num,
        batch_size=_BATCH_SIZE,
        face_modality=_FACE_MODALITY,
        is_training=False
    )

    frame_test_data_generator = load_dataset_features.DataGeneratorFaceFrame(
        data_folder=data_dir,
        test_actors=test_actors,
        frame_num=None,
        batch_size=_BATCH_SIZE,
        face_modality=_FACE_MODALITY,
        is_training=False
    )

    return {
        "cnn": cnn_test_data_generator,
        "cnn_lip": cnn_lip_test_data_generator,
        "rnn": rnn_test_data_generator,
        "rnn_lip": rnn_lip_test_data_generator,
        "frame": frame_test_data_generator
    }


def load_training_log_df(training_log_path: str):
    df = pandas.read_csv(training_log_path, sep=",")
    return df


def load_model_weights(weights_path):
    model = keras.models.load_model(weights_path)
    return model


def extract_data_from_generator(generator):
    generator_len = len(generator)

    inputs = list()
    labels = list()
    metadata = list()

    for x in range(generator_len):
        _in_data, _label = generator[x]
        inputs.extend(list(_in_data))
        labels.extend(list(_label))
        metadata.extend(generator.get_last_data_info())

    return numpy.array(inputs), numpy.array(labels), metadata


def predict(model, inputs):
    return model.predict_generator(inputs)


def get_accuracy_and_conf_matrix(ground_truth, predictions):
    predicted_class = keras.backend.argmax(predictions)
    accuracy = metrics.accuracy_score(
        ground_truth, predicted_class, normalize=True
    )
    conf_matrix = metrics.confusion_matrix(
        ground_truth, predicted_class, normalize="true"
    )

    return accuracy, conf_matrix


def get_accuracy_and_conf_matrix_frame(ground_truth, predictions):
    accuracy = metrics.accuracy_score(
        ground_truth, predictions, normalize=True
    )
    conf_matrix = metrics.confusion_matrix(
        ground_truth, predictions, normalize="true"
    )

    return accuracy, conf_matrix


def test_frame_model(data_dir, test_actors, frames_list, model, training_log):
    logger = logging.getLogger(f"{__name__}.{test_frame_model.__name__}")
    results = list()

    performances = list()
    for num_frames in frames_list:
        frame_test_data_generator = load_dataset_features.DataGeneratorFaceRNN(
            data_folder=data_dir,
            test_actors=test_actors,
            frame_num=num_frames,
            batch_size=1,
            face_modality=_FACE_MODALITY,
            is_training=False
        )
        inputs_list, labels, metadata = extract_data_from_generator(frame_test_data_generator)

        flattened_inputs = inputs_list.reshape((inputs_list.shape[0] * inputs_list.shape[1], inputs_list.shape[2]))
        predictions = predict(model, flattened_inputs)
        predictions = predictions.reshape((inputs_list.shape[0], inputs_list.shape[1], 7))

        predictions = numpy.argmax(predictions, axis=2)
        predictions = numpy.array(stats.mode(predictions, axis=1)[0])
        predictions = numpy.squeeze(predictions)

        performances.append({
            "model": "frame",
            "num_frames": num_frames,
            "labels": labels,
            "predictions": predictions,
            "metadata": metadata
        })

        accuracy, conf_matrix = get_accuracy_and_conf_matrix_frame(labels, predictions)

        logger.info(f"tested frame model at {num_frames} frames with accuracy {accuracy}")
        results.append({
            "name": "frame",
            "frames": num_frames,
            "accuracy": accuracy,
            "conf_matrix": conf_matrix,
            "training_log": training_log
        })

    return results, performances


def main(config):
    logger = logging.getLogger(f"{__name__}.{main.__name__}")

    test_stats = list()
    performances = list()

    # RNN and CNN models
    logger.info(f"evaluating RNN and CNN models for frames {config.frames_list}")
    for num_frames in config.frames_list:
        logger.info(f"getting model weights and stats paths from {config.model_dir} for frame number {num_frames}")
        model_weights_stats_paths = get_model_weights_and_stats_paths(model_dir=config.model_dir, num_frames=num_frames)

        logger.info(f"constructing test data generators for frame number {num_frames}")
        test_data_generators = construct_test_data_generators(
            data_dir=config.data_dir, test_actors=config.test_actors, frame_num=num_frames
        )

        model_names = ["cnn", "cnn_lip", "rnn", "rnn_lip"]
        for model_name in model_names:
            logger.info(f"processing {model_name.upper()} model")
            model_weights_path, model_logs_path = model_weights_stats_paths[model_name]
            test_data_generator = test_data_generators[model_name]

            model = load_model_weights(model_weights_path)
            training_log = load_training_log_df(model_logs_path)

            inputs, labels, metadata = extract_data_from_generator(test_data_generator)
            predictions = predict(model, inputs)

            performances.append({
                "model": model_name,
                "num_frames": num_frames,
                "labels": labels,
                "metadata": metadata,
                "predictions": predictions
            })


            accuracy, conf_matrix = get_accuracy_and_conf_matrix(labels, predictions)

            logger.info(f"tested {model_name} at {num_frames} frames with accuracy {accuracy}")
            test_stats.append({
                "name": model_name,
                "frames": num_frames,
                "accuracy": accuracy,
                "conf_matrix": conf_matrix,
                "training_log": training_log
            })

    # frame model
    logger.info(f"evaluating frame model")
    frame_model_weights_path, frame_model_log_path = get_frame_model_weights_and_stats_path(config.model_dir)
    frame_model = load_model_weights(frame_model_weights_path)
    training_log = load_training_log_df(frame_model_log_path)

    results, p = test_frame_model(
        data_dir=config.data_dir,
        test_actors=config.test_actors,
        frames_list=config.frames_list,
        model=frame_model,
        training_log=training_log
    )
    performances.extend(p)
    test_stats.extend(results)

    logger.info(f"saving to {config.save_path}")
    with open(config.save_path, "wb") as ostream:
        pickle.dump(test_stats, ostream)

    with open(config.performances_save_path, "wb") as ostream:
        pickle.dump(performances, ostream)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-c", "--config_dir", help="path to the dir containing configuration files"
    )
    args = arg_parser.parse_args()

    setup_logging(
        os.path.join(args.config_dir, "logging_config.yml")
    )
    training_config_path = Configuration.load_from_yaml_file(
        os.path.join(args.config_dir, "generate_test_stats_config.yml")
    )

    main(config=training_config_path)
