import argparse
import logging
import os

from tensorflow import keras

from preprocess import load_dataset_features
from util.util import Configuration
from util.logging_setup import setup_logging


def main(config):
    _logger = logging.getLogger(f"{__name__}.{main.__name__}")

    _logger.info("creating training data generator")
    training_data_generator = load_dataset_features.DataGeneratorFaceFrame(
        data_folder=config.data_dir,
        test_actors=config.test_actors,
        frame_num=None,
        batch_size=config.batch_size,
        face_modality=config.face_modality,
        is_training=True
    )
    _logger.info("creating validation data generator")
    validation_data_generator = load_dataset_features.DataGeneratorFaceFrame(
        data_folder=config.data_dir,
        test_actors=config.test_actors,
        frame_num=None,
        batch_size=config.batch_size,
        face_modality=config.face_modality,
        is_training=False
    )

    _logger.info("declaring keras model")
    input_shape = (
        config.feature_len
    )
    input_layer = keras.layers.Input(shape=input_shape)
    hidden_layer = keras.layers.Dense(256, activation="relu")(input_layer)
    dropout_hidden = keras.layers.Dropout(rate=config.dense_dropout)(hidden_layer)
    output_layer = keras.layers.Dense(config.num_classes, activation="softmax")(dropout_hidden)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    _logger.info("compiling keras model")
    optmizer = keras.optimizers.Adam(learning_rate=config.learning_rate)
    loss = keras.losses.SparseCategoricalCrossentropy()
    model.compile(
        optimizer=optmizer,
        loss=loss,
        metrics=["accuracy"]
    )

    print(model.summary())

    _logger.info("fitting data to model")
    csv_logger = keras.callbacks.CSVLogger(config.logging_output, separator=",")
    model.fit(
        x=training_data_generator,
        validation_data=validation_data_generator,
        epochs=config.epochs,
        callbacks=[csv_logger]
    )

    _logger.info(f"saving model to {config.model_save_path}")
    model.save(config.model_save_path)


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
        os.path.join(args.config_dir, "fer_speaking_effect_frame_train_config.yml")
    )

    main(config=training_config_path)
