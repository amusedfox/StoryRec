import os
from pathlib import Path

import plac
import sys

sys.path.append('.')
from story.base import ILLEGAL_FILE_CHARS


@plac.pos('dir_to_search', "Directory to search", Path)
def main(dir_to_search):
    for root, dirs, files in os.walk(dir_to_search):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            new_file_name = ILLEGAL_FILE_CHARS.sub("", file_name)
            if file_name != new_file_name:
                os.rename(file_path, os.path.join(root, new_file_name))
                print(file_name)
                print(new_file_name)


if __name__ == '__main__':
    plac.call(main)
