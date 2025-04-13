"""
Facial Expression Recognition 3D CNN model with Speaking Effect
"""

from tensorflow import keras


class SptioTemporalRNN(object):
    def __init__(self, num_frames, feature_length, num_channels, num_classes, gru_dropout_rate, learning_rate):
        self._num_frames = num_frames
        self._feature_length = feature_length
        self._num_channels = num_channels
        self._num_classes = num_classes
        self._gru_dropout_rate = gru_dropout_rate
        self._learning_rate = learning_rate

    def construct_model(self):
        # define input layer
        input_shape = (
            self._num_frames, self._feature_length
        )
        input_layer = keras.layers.Input(shape=input_shape)

        gru_layer_1 = keras.layers.Bidirectional(
            keras.layers.GRU(units=8)
        )(input_layer)
        dropout_layer_1 = keras.layers.Dropout(rate=self._gru_dropout_rate)(gru_layer_1)

        dense_layer_1 = keras.layers.Dense(16)(dropout_layer_1)

        output_layer = keras.layers.Dense(
            units=self._num_classes,
            activation="softmax"
        )(dense_layer_1)

        self._model = keras.Model(inputs=input_layer, outputs=output_layer)
        return self

    def train_model(self, train_generator, valid_generator, epochs, logging_output):
        optmizer = keras.optimizers.Adam(learning_rate=self._learning_rate)
        loss = keras.losses.SparseCategoricalCrossentropy()
        self._model.compile(
            optimizer=optmizer,
            loss=loss,
            metrics=["accuracy"]
        )

        csv_logger = keras.callbacks.CSVLogger(logging_output, separator=",")
        self._model.fit(
            x=train_generator,
            validation_data=valid_generator,
            epochs=epochs,
            callbacks=[csv_logger]
        )

    @property
    def model(self):
        return self._model
