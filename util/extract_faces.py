import argparse
import logging
import os

import cv2
import mtcnn
from PIL import Image

from util.util import get_filename_from_path_without_extension
from util.logging_setup import setup_logging


class ImageFaceExtractor(object):
    def __init__(self):
        self._detector = mtcnn.MTCNN()
        self._logger = logging.getLogger(ImageFaceExtractor.__name__)

    def crop_face_from_img(self, cv2_img, face_box):
        x, y, width, height = face_box
        x, y = max(x, 0), max(y, 0)

        return cv2_img[y: y + height, x: x + width].copy()

    def from_cv2_img(self, cv2_img):
        detected_faces = self._detector.detect_faces(cv2_img)
        self._logger.debug(f"detected {len(detected_faces)} faces")

        cropped_faces = list()
        for idx, face_metadata in enumerate(detected_faces):
            face_box = face_metadata["box"]
            self._logger.debug(
                f"face box {idx}: x: {face_box[0]}; "
                f"y:{face_box[1]}; width: {face_box[2]}; "
                f"height: {face_box[3]}"
            )
            self._logger.debug("confidence: {}".format(face_metadata["confidence"]))

            cropped_face_img = self.crop_face_from_img(cv2_img, face_box)
            cropped_faces.append(cropped_face_img)

        return cropped_faces


@DeprecationWarning
class CroppedFace(object):
    def __init__(self, cropped_face, video_name, frame_num, face_num):
        self._cropped_face = cropped_face
        self._video_name = video_name
        self._frame_num = frame_num
        self._face_num = face_num

        self._emotion_value = None

    def has_face(self):
        return len(self._cropped_face) > 0

    def to_dict(self):
        return {
            "video_name": self._video_name,
            "frame_num": self._frame_num,
            "face_num": self._face_num,
            "emo_val": self._emotion_value
        }

    @property
    def cropped_face_cv2(self):
        return self._cropped_face

    @property
    def cropped_face_pillow(self):
        img = cv2.cvtColor(self._cropped_face, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    @property
    def video_name(self):
        return self._video_name

    @property
    def frame_num(self):
        return self._frame_num

    @property
    def face_num(self):
        return self._face_num

    @property
    def emotion_value(self):
        return self._emotion_value

    @emotion_value.setter
    def emotion_value(self, value):
        self._emotion_value = value


class VideoFaceExtractor(object):
    def __init__(self, output_extension="jpg"):
        self._image_face_extractor = ImageFaceExtractor()
        self._output_extension = output_extension

        # uncomment for use
        self._logger = logging.getLogger(VideoFaceExtractor.__name__)

    # def process_video(self, video_path, video_name, output_folder):
    #     video_capture = cv2.VideoCapture(video_path)
    #     success, image = video_capture.read()
    #     print(f"reading video \"{video_name}\"")

    #     frame_count = 0
    #     while success:
    #         cropped_faces = self._image_face_extractor.from_cv2_img(image)
    #         out_template = os.path.join(output_folder, f"{video_name}_frame_{frame_count}_face_")
    #         for idx, cropped_face_img in enumerate(cropped_faces):
    #             out_path = out_template + f"{idx}" + "." + self._output_extension
    #             cv2.imwrite(out_path, cropped_face_img)
    #         print(
    #             f"detected {len(cropped_faces)} faces - saved to "
    #             f"{out_template}num.{self._output_extension}",
    #             end="\r", flush=True
    #         )

    #         success, image = video_capture.read()
    #         frame_count += 1

    #     video_capture.release()

    def get_cropped_faces(self, video_path, video_name):
        video_capture = cv2.VideoCapture(video_path)
        success, image = video_capture.read()

        frame_count = 0
        cropped_faces_list = list()
        while success:
            cropped_faces = self._image_face_extractor.from_cv2_img(image)
            self._logger.debug(
                f"processing frame {frame_count} for {video_name};"
                f" found {len(cropped_faces)} faces"
            )
            for idx, cropped_face_img in enumerate(cropped_faces):
                cropped_faces_list.append(
                    CroppedFace(
                        cropped_face=cropped_face_img,
                        video_name=video_name,
                        frame_num=frame_count,
                        face_num=idx
                    )
                )

            success, image = video_capture.read()
            frame_count += 1

        video_capture.release()

        return cropped_faces_list


def _test(args):
    video_face_extractor = VideoFaceExtractor(args.ext)
    video_name = get_filename_from_path_without_extension(args.vid)

    video_face_extractor.process_video(args.vid, video_name, args.out_dir)


if __name__ == "__main__":
    setup_logging("config/logging_config.yml")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--vid", required=True)
    arg_parser.add_argument("--out_dir", required=True)
    arg_parser.add_argument("--ext", required=True)
    args = arg_parser.parse_args()

    _test(args)
