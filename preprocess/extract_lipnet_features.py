import argparse
import logging
import os
import pickle

from lipnet.lip_reading_feature_extractor import LipnetFeatureExtractor
from util.logging_setup import setup_logging
from util.ravdess import parse_video_metadata
from util.util import get_files_recursively_ext, Configuration, remove_file, \
    get_filename_from_path_without_extension, change_video_fps


def skip_video(video_name: str) -> bool:
    video_metadata = parse_video_metadata(video_name)
    # skip if it's not video-only
    if video_metadata["modality"] != 2:
        return True
    # skip if the emotion is "calm"
    if video_metadata["emotion"] == 2:
        return True
    return False


def main(lipnet_config: Configuration, subdir: str) -> None:
    logger = logging.getLogger(main.__name__)
    logger.debug("debug test")

    videos_path = os.path.join(lipnet_config.video_folder, subdir)
    all_videos_list = get_files_recursively_ext(
        videos_path, lipnet_config.video_extensions
    )
    logger.info("found {} video files in {} with extensions {}".format(
        len(all_videos_list), videos_path,
        lipnet_config.video_extensions
    ))

    lipnet_feature_extractor = LipnetFeatureExtractor(lipnet_config)

    video_features = dict()
    num_videos = len(all_videos_list)
    for idx, video_path in enumerate(all_videos_list):
        video_name = get_filename_from_path_without_extension(video_path)
        if skip_video(video_name):
            logger.info(f"skipping video {idx + 1}/{num_videos}; path {video_path}")
            continue

        temp_video_file = change_video_fps(video_path, lipnet_config.temp_folder, lipnet_config.dest_frame_rate)

        logger.info(f"processing video {idx + 1}/{num_videos}; path {video_path}")

        lipnet_features = lipnet_feature_extractor.calculate_predictions(temp_video_file)
        video_features[video_name] = lipnet_features

        remove_file(temp_video_file)

    save_path = lipnet_config.save_path.format(subdir)
    logger.info(f"saving video emotions to {save_path}")
    with open(save_path, "wb") as ostream:
        pickle.dump(
            video_features, ostream
        )

    logger.info("done")


if __name__ == '__main__':
    # choose a particular device on machine
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-c", "--config_dir", required=True,
        help="path to the directory containing config files"
    )
    arg_parser.add_argument(
        "-s", "--subdir", required=True,
        help="The subdir to the main data dir to elaborate"
    )
    args = arg_parser.parse_args()

    setup_logging(
        os.path.join(args.config_dir, "logging_config.yml")
    )

    lipnet_config = Configuration.load_from_yaml_file(
        os.path.join(args.config_dir, "lipnet_config.yml")
    )

    main(lipnet_config=lipnet_config, subdir=args.subdir)
