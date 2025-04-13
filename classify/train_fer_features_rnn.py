import argparse
import logging
import os

from models import fer_speaking_effect_rnn
from preprocess import load_dataset_features
from util.util import Configuration
from util.logging_setup import setup_logging


def main(config):
    _logger = logging.getLogger(f"{__name__}.{main.__name__}")

    _logger.info("creating training data generator")
    training_data_generator = load_dataset_features.DataGeneratorFaceRNN(
        data_folder=config.data_dir,
        test_actors=config.test_actors,
        frame_num=config.num_frames,
        batch_size=config.batch_size,
        face_modality=config.face_modality,
        is_training=True
    )
    _logger.info("creating validation data generator")
    validation_data_generator = load_dataset_features.DataGeneratorFaceRNN(
        data_folder=config.data_dir,
        test_actors=config.test_actors,
        frame_num=config.num_frames,
        batch_size=config.batch_size,
        face_modality=config.face_modality,
        is_training=False
    )

    _logger.info("creating the model")
    fer_cnn = fer_speaking_effect_rnn.SptioTemporalRNN(
        num_frames=config.num_frames,
        feature_length=config.feature_len,
        num_channels=config.num_channels,
        num_classes=config.num_classes,
        gru_dropout_rate=config.gru_dropout_rate,
        learning_rate=config.learning_rate
    ).construct_model()

    _logger.info("initiating training")
    fer_cnn.train_model(
        train_generator=training_data_generator,
        valid_generator=validation_data_generator,
        epochs=config.epochs,
        logging_output=config.logging_output
    )

    _logger.info(f"saving model to {config.model_save_path}")
    fer_cnn.model.save(config.model_save_path)


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
        os.path.join(args.config_dir, "fer_speaking_effect_rnn_train_config.yml")
    )

    main(config=training_config_path)
