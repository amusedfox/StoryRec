from bs4 import BeautifulSoup
import os
import re

SPECIAL_CHARS = re.compile(r'[<>:"\\/|*\n]')

for subdir, dirs, files in os.walk('../Stories'):
    for file_name in files:
        filepath = os.path.join(subdir, file_name)

        if not filepath.endswith(".html"):
            continue

        new_file_name = SPECIAL_CHARS.sub("", file_name)
        if file_name != new_file_name:
            os.rename(filepath, os.path.join(subdir, new_file_name))
            print(file_name)
            print(new_file_name)

