from tensorflow import keras
from IPython.display import SVG
from IPython.display import Latex

from models import fer_speaking_effect_cnn, fer_speaking_effect_rnn
from util.util import Configuration

CNN_CONFIG_PATH = "config/fer_speaking_effect_cnn_train_config_lip.yml"
RNN_CONFIG_PATH = "config/fer_speaking_effect_rnn_train_config_lip.yml"

CNN_PLOT_OUTPUT = "cnn_plot.pdf"
RNN_PLOT_OUTPUT = "rnn_plot.pdf"


cnn_config = Configuration.load_from_yaml_file(file_path=CNN_CONFIG_PATH)
rnn_config = Configuration.load_from_yaml_file(file_path=RNN_CONFIG_PATH)


st_cnn = fer_speaking_effect_cnn.SptioTemporalCNN(
    num_frames=cnn_config.num_frames,
    feature_length=cnn_config.feature_len,
    num_channels=cnn_config.num_channels,
    num_classes=cnn_config.num_classes,
    conv_dropout_rate=cnn_config.conv_dropout_rate,
    learning_rate=cnn_config.learning_rate
).construct_model()


rnn = fer_speaking_effect_rnn.SptioTemporalRNN(
    num_frames=rnn_config.num_frames,
    feature_length=rnn_config.feature_len,
    num_channels=rnn_config.num_channels,
    num_classes=rnn_config.num_classes,
    gru_dropout_rate=rnn_config.gru_dropout_rate,
    learning_rate=rnn_config.learning_rate
).construct_model()


# keras.utils.plot_model(st_cnn.model, to_file=CNN_PLOT_OUTPUT, show_shapes=False, expand_nested=True)
# keras.utils.plot_model(rnn.model, to_file=RNN_PLOT_OUTPUT, show_shapes=False, expand_nested=True)

cnn_dot = keras.utils.model_to_dot(st_cnn.model, show_shapes=False, show_layer_names=False, expand_nested=True)
rnn_dot = keras.utils.model_to_dot(rnn.model, show_layer_names=False, show_shapes=False, expand_nested=True)

cnn_dot.write(path=CNN_PLOT_OUTPUT, prog='dot', format='pdf')
rnn_dot.write(path=RNN_PLOT_OUTPUT, prog='dot', format='pdf')

