import string
import os
from pathlib import Path

from tqdm import tqdm
import plac


@plac.pos('dir_to_fix', 'Directory to fix', type=Path)
@plac.pos('html_dir', 'Directory with html files', type=Path)
@plac.pos('txt_dir', 'Directory with text files', type=Path)
def main(dir_to_fix, html_dir, txt_dir):
    """Fix story directory structure

    Since there are too many files under one directory, choose subdirectories
    for each file using the first letter of the filename.

    Also separate .txt files from .html files to another directory.
    """

    for char in string.ascii_lowercase:
        os.makedirs(os.path.join(html_dir, char), exist_ok=True)
        os.makedirs(os.path.join(txt_dir, char), exist_ok=True)

    for subdir, dirs, files in tqdm(list(os.walk(dir_to_fix))):

        if subdir == str(dir_to_fix):
            continue

        for file_name in tqdm(files):

            # Get first char that is a letter
            first_char = file_name[0]
            first_char_i = 1
            while not first_char.isalpha():
                first_char = file_name[first_char_i]
                first_char_i += 1
            first_char = first_char.lower()

            if file_name.endswith('.html'):
                new_file_path = os.path.join(html_dir, first_char, file_name)
            elif file_name.endswith('.txt'):
                new_file_path = os.path.join(txt_dir, first_char, file_name)
            else:
                continue

            file_path = os.path.join(subdir, file_name)
            os.rename(file_path, new_file_path)


if __name__ == '__main__':
    plac.call(main)
