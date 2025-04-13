import os
import subprocess
import sys

import yaml


class Configuration(object):
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

    @staticmethod
    def load_from_yaml_file(file_path):
        with open(file_path, "r") as istream:
            yaml_dict = yaml.load(istream, Loader=yaml.Loader)
        return Configuration(yaml_dict)


def get_files_recursively_ext(path: str, extensions):
    files = list()
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            is_of_supported_format = any([
                file.endswith(extension) for extension in extensions
            ])
            if is_of_supported_format:
                files.append(os.path.join(r, file))

    return files


def find_files_in_folder_by_format(folder_path, formats_supported):
    file_list = list()
    formats_supported = [
        file_format[1:] if file_format[0] == "." else file_format
        for file_format in formats_supported
    ]
    format_file_endings = ["." + file_format for file_format in formats_supported]

    for file in os.listdir(folder_path):
        is_of_supported_format = any([
            file.endswith(format_ending) for format_ending in format_file_endings
        ])
        if is_of_supported_format:
            full_path = os.path.join(folder_path, file)
            file_list.append(full_path)

    return file_list


def run_command(command, cwd=None, capture_stdout=False):
    if capture_stdout:
        process = subprocess.Popen(
            command, shell=False, stderr=sys.stderr, stdout=subprocess.PIPE, stdin=sys.stdin, cwd=cwd
        )
        process.wait()
        # return only stdout and convert the bytes to string
        return process.communicate()[0].decode("utf-8")
    else:
        process = subprocess.Popen(
            command, shell=False, stderr=sys.stderr, stdout=sys.stdout, stdin=sys.stdin, cwd=cwd
        )
        process.wait()


def remove_file(file_path):
    os.unlink(file_path)


def get_filename_from_path_without_extension(path: str) -> str:
    return str(os.path.basename(path).rsplit(".", 1)[0])


def change_video_fps(video_path, temp_dir, dest_frame_rate):
    file_name = os.path.split(video_path)[1]
    dest_path = os.path.join(temp_dir, file_name)

    command = f"ffmpeg -y -i {video_path} -filter:v fps=fps={dest_frame_rate} -strict -2 {dest_path}"
    command_lst = command.split()
    run_command(command_lst, capture_stdout=True)

    return dest_path
