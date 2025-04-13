import argparse
import logging
import os
import pickle

import cv2

from preprocess import face_tracking
from util.logging_setup import setup_logging
from util.ravdess import parse_video_metadata
from util.util import Configuration, get_files_recursively_ext, remove_file, get_filename_from_path_without_extension, \
    change_video_fps

__ALL_VIDEOS = "__all__"
_logger = logging.getLogger(__name__)


def get_video_face_crops(video_file: str, video_face_extractor, input_dims):
    _logger.info("detecting and cropping faces")
    face_crops = video_face_extractor.detect_faces_in_video(video_file)
    resized_face_crops = [
        cv2.resize(face_crop, tuple(input_dims)) for face_crop in face_crops
    ]

    return resized_face_crops


def skip_video(video_name: str) -> bool:
    video_metadata = parse_video_metadata(video_name)
    # skip if it's not video-only
    if video_metadata["modality"] != 2:
        return True
    # skip if the emotion is "calm"
    if video_metadata["emotion"] == 2:
        return True
    return False


def main(extract_face_frames_config, subdir):
    logger = logging.getLogger(main.__name__)

    if subdir == __ALL_VIDEOS:
        root_video_path = extract_face_frames_config.video_folder
    else:
        root_video_path = os.path.join(extract_face_frames_config.video_folder, subdir)
    all_videos_list = get_files_recursively_ext(
        root_video_path, extract_face_frames_config.video_extensions
    )
    logger.info(
        f"found {len(all_videos_list)} video files "
        f"in {root_video_path} "
        f"with extensions {extract_face_frames_config.video_extensions}"
    )

    video_face_extractor = face_tracking.VideoFaceDetectorKalman(
        downsample_rate=extract_face_frames_config.downsample_rate
    )

    video_face_frames = dict()
    num_videos = len(all_videos_list)
    for idx, video_path in enumerate(all_videos_list):
        video_name = get_filename_from_path_without_extension(video_path)
        if skip_video(video_name):
            logger.info(f"skipping video {idx + 1}/{num_videos}; path {video_path}")
            continue

        temp_video_file = change_video_fps(video_path, extract_face_frames_config.temp_folder,
                                           extract_face_frames_config.dest_frame_rate)

        logger.info(f"processing video {idx + 1}/{num_videos}; path {video_path}")
        video_face_features = get_video_face_crops(
            video_file=video_path,
            video_face_extractor=video_face_extractor,
            input_dims=extract_face_frames_config.input_dims
        )
        video_face_frames[video_name] = video_face_features

        remove_file(temp_video_file)

    if subdir == __ALL_VIDEOS:
        save_path = extract_face_frames_config.save_path.format("all")
    else:
        actor_num = int(subdir.split("_")[1])
        save_path = extract_face_frames_config.save_path.format(actor_num)
    logger.info(f"saving video emotions to {save_path}")
    with open(save_path, "wb") as ostream:
        pickle.dump(video_face_frames, ostream)

    logger.info("done")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-c", "--config_dir", required=True,
        help="path to the directory containing config files"
    )
    arg_parser.add_argument(
        "-s", "--subdir", required=False, default=__ALL_VIDEOS,
        help="The subdir to the main data dir to elaborate"
    )
    args = arg_parser.parse_args()

    setup_logging(
        os.path.join(args.config_dir, "logging_config.yml")
    )

    extract_face_frames_config = Configuration.load_from_yaml_file(
        os.path.join(args.config_dir, "extract_face_frames_config.yml")
    )

    main(
        extract_face_frames_config=extract_face_frames_config, subdir=args.subdir
    )
