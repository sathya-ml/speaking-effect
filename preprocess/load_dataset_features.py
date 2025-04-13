import logging
import pickle

import numpy
from tensorflow import keras

from util.ravdess import parse_video_metadata
from util.util import find_files_in_folder_by_format, get_filename_from_path_without_extension

_logger = logging.getLogger(__name__)


class Dataload(object):
    @staticmethod
    def _load_vgg_face_features(data_folder: str):
        all_feature_files = find_files_in_folder_by_format(data_folder, ["pkl"])
        vgg_face_features_file = [
            file for file in all_feature_files
            if "vgg_face" in get_filename_from_path_without_extension(file)
        ][0]

        with open(vgg_face_features_file, "rb") as istream:
            return pickle.load(istream)

    @staticmethod
    def _load_vgg_emotion_features(data_folder: str):
        all_feature_files = find_files_in_folder_by_format(data_folder, ["pkl"])
        vgg_emotion_features_file = [
            file for file in all_feature_files
            if "vgg_emotion" in get_filename_from_path_without_extension(file)
        ][0]

        with open(vgg_emotion_features_file, "rb") as istream:
            return pickle.load(istream)

    @staticmethod
    def _load_lipnet_features(data_folder: str):
        all_feature_files = find_files_in_folder_by_format(data_folder, ["pkl"])
        lipnet_feature_files = [
            file for file in all_feature_files
            if "lipnet" in get_filename_from_path_without_extension(file)
        ]

        data_dict = dict()
        for lipnet_file in lipnet_feature_files:
            with open(lipnet_file, "rb") as istream:
                data_dict.update(pickle.load(istream))

        return data_dict

    @staticmethod
    def _merge_features(vgg_emo_features, vgg_face_features, lipnet_features):
        composite_feature_dict = dict()
        all_keys = list(vgg_emo_features.keys())

        for key in all_keys:
            face = numpy.array(vgg_face_features[key])
            emo = numpy.array(vgg_emo_features[key])
            lipnet = numpy.squeeze(lipnet_features[key])

            if face.shape[0] != emo.shape[0] or face.shape[0] != lipnet.shape[0]:
                min_len = min((face.shape[0], emo.shape[0], lipnet.shape[0]))
                _logger.warning(
                    f"mismatched shapes: face -> {face.shape[0]}; "
                    f"emo -> {emo.shape[0]}; lipnet -> {lipnet.shape[0]}"
                )
            else:
                min_len = face.shape[0]

            composite_feature_dict[key] = (
                face[:min_len, ...], emo[:min_len, ...], lipnet[:min_len, ...]
            )

        return composite_feature_dict

    @staticmethod
    def _clean_dict_keys(data_dict):
        """
        This part has to be adapted to each particular preprocessing procedure
        """
        clean_data_list = list()
        for key, val in data_dict.items():
            clean_data_list.append({
                "data": val
            })
            metadata = parse_video_metadata(key)
            clean_data_list[-1].update(metadata)
            clean_data_list[-1].update({"video_name": key})

        return clean_data_list

    @staticmethod
    def load_data(data_folder: str):
        lipnet_features = Dataload._load_lipnet_features(data_folder)
        vgg_face_features = Dataload._load_vgg_face_features(data_folder)
        vgg_emo_features = Dataload._load_vgg_emotion_features(data_folder)

        merged_data = Dataload._merge_features(
            vgg_emo_features=vgg_emo_features, vgg_face_features=vgg_face_features, lipnet_features=lipnet_features
        )

        clean_data = Dataload._clean_dict_keys(merged_data)

        return clean_data


def exclude_class(data_list, class_to_exclude):
    data_list = [
        data for data in data_list
        if data["emotion"] != class_to_exclude
    ]
    # correct the other indices
    for data in data_list:
        if data["emotion"] > class_to_exclude:
            data["emotion"] = data["emotion"] - 1
    return data_list


def exclude_data_with_property(data_list, property_name, val_to_exclude):
    data_list = [
        data for data in data_list
        if data[property_name] != val_to_exclude
    ]
    return data_list


def separate_test_and_train(full_data, test_data_actors_list):
    train_data = [
        video_data for video_data in full_data
        if video_data["actor_num"] not in test_data_actors_list
    ]
    test_data = [
        video_data for video_data in full_data
        if video_data["actor_num"] in test_data_actors_list
    ]

    return train_data, test_data


def preprocess_data(data_dict) -> list:
    # exclude the calm class
    data = exclude_class(data_dict, class_to_exclude=2)
    return data


class DataGeneratorBase(keras.utils.Sequence):
    def __init__(self, data_folder, test_actors, frame_num, batch_size, face_modality, is_training=True):
        self._data_folder = data_folder
        self._frame_num = frame_num
        self._batch_size = batch_size
        self._test_actors = test_actors
        self._is_training = is_training
        self._face_modality = face_modality

        data_dict = Dataload.load_data(self._data_folder)
        data = preprocess_data(data_dict)

        if self._is_training:
            self._data = separate_test_and_train(data, self._test_actors)[0]
        else:
            self._data = separate_test_and_train(data, self._test_actors)[1]

        self._last_start_frame = None
        self._last_end_frame = None
        self._last_video_name = None
        self._last_data_infos = list()

    def __len__(self):
        instances_count = 0
        for video in self._data:
            video_len = video["data"][0].shape[0]
            instances_count += video_len - self._frame_num + 1
        return int(instances_count / self._batch_size)

    def get_last_data_info(self) -> list:
        return self._last_data_infos

    def _get_rand_data_slice_to_append(self):
        video_num = numpy.random.randint(len(self._data))
        video_len = self._data[video_num]["data"][0].shape[0]

        if video_len < self._frame_num:
            _logger.error(f"video length less than frame_num: {video_len} < {self._frame_num}")
            raise AssertionError

        start_idx_num = numpy.random.randint(video_len - self._frame_num)
        end_idx_num = start_idx_num + self._frame_num

        self._last_start_frame = start_idx_num
        self._last_end_frame = end_idx_num
        self._last_video_name = self._data[video_num]["video_name"]

        if self._face_modality == "face":
            data_slice_to_append_face = self._data[video_num]["data"][0][start_idx_num: end_idx_num, ...]
        elif self._face_modality == "emotion":
            data_slice_to_append_face = self._data[video_num]["data"][1][start_idx_num: end_idx_num, ...]
        else:
            raise AssertionError(f"modality \"{self._face_modality}\" not recognized")

        data_slice_to_append_lip = self._data[video_num]["data"][2][start_idx_num: end_idx_num, ...]
        label = self._data[video_num]["emotion"] - 1

        return data_slice_to_append_face, data_slice_to_append_lip, label


class DataGeneratorFaceCNN(DataGeneratorBase):
    def __getitem__(self, index):
        data = list()
        labels = list()
        self._last_data_infos = list()

        for _ in range(self._batch_size):
            data_slice_to_append, _, label = self._get_rand_data_slice_to_append()

            data.append(
                numpy.expand_dims(
                    data_slice_to_append, axis=0
                )
            )
            labels.append(label)
            self._last_data_infos.append({
                "start_frame": self._last_start_frame,
                "end_frame": self._last_end_frame,
                "video_name": self._last_video_name
            })

        data, labels = numpy.array(data), numpy.array(labels)
        _logger.debug(f"data shape: {data.shape}")

        return data, labels


class DataGeneratorFaceRNN(DataGeneratorBase):
    def __getitem__(self, index):
        data = list()
        labels = list()
        self._last_data_infos = list()

        for _ in range(self._batch_size):
            data_slice_to_append, _, label = self._get_rand_data_slice_to_append()

            data.append(data_slice_to_append)
            labels.append(label)
            self._last_data_infos.append({
                "start_frame": self._last_start_frame,
                "end_frame": self._last_end_frame,
                "video_name": self._last_video_name
            })

        data, labels = numpy.array(data), numpy.array(labels)
        _logger.debug(f"data shape: {data.shape}")

        return data, labels


class DataGeneratorLipCNN(DataGeneratorBase):
    def __getitem__(self, index):
        data = list()
        labels = list()
        self._last_data_infos = list()

        for _ in range(self._batch_size):
            data_slice_to_append_face, data_slice_to_append_lip, label = self._get_rand_data_slice_to_append()
            data_slice_to_append = numpy.concatenate(
                (data_slice_to_append_face, data_slice_to_append_lip), axis=1
            )

            data.append(
                numpy.expand_dims(
                    data_slice_to_append, axis=0
                )
            )
            labels.append(label)
            self._last_data_infos.append({
                "start_frame": self._last_start_frame,
                "end_frame": self._last_end_frame,
                "video_name": self._last_video_name
            })

        data, labels = numpy.array(data), numpy.array(labels)
        _logger.debug(f"data shape: {data.shape}")

        return data, labels


class DataGeneratorLipRNN(DataGeneratorBase):
    def __getitem__(self, index):
        data = list()
        labels = list()
        self._last_data_infos = list()

        for _ in range(self._batch_size):
            data_slice_to_append_face, data_slice_to_append_lip, label = self._get_rand_data_slice_to_append()
            data_slice_to_append = numpy.concatenate(
                (data_slice_to_append_face, data_slice_to_append_lip), axis=1
            )

            data.append(data_slice_to_append)
            labels.append(label)
            self._last_data_infos.append({
                "start_frame": self._last_start_frame,
                "end_frame": self._last_end_frame,
                "video_name": self._last_video_name
            })

        data, labels = numpy.array(data), numpy.array(labels)
        _logger.debug(f"data shape: {data.shape}")

        return data, labels


class DataGeneratorFaceFrame(DataGeneratorBase):
    def __init__(self, data_folder, test_actors, frame_num, batch_size, face_modality, is_training=True):
        super(DataGeneratorFaceFrame, self).__init__(
            data_folder, test_actors, frame_num, batch_size, face_modality, is_training
        )
        self._prepare_data()

    def _prepare_data(self):
        self._data_x, self._data_y = list(), list()
        for data in self._data:
            for i in range(data["data"][1].shape[0]):
                self._data_x.append(
                    data["data"][1][i, ...]
                )
                self._data_y.append(data["emotion"] - 1)

    def __len__(self):
        return int(len(self._data_x) / self._batch_size)

    def __getitem__(self, index):
        data = list()
        labels = list()
        self._last_data_infos = list()

        indices = numpy.random.choice(len(self._data_x), self._batch_size)
        for idx in indices:
            data.append(self._data_x[idx])
            labels.append(self._data_y[idx])

        data, labels = numpy.array(data), numpy.array(labels)
        _logger.debug(f"data shape: {data.shape}")

        return data, labels
