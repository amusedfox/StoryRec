import os
import sys
from tqdm.auto import tqdm

for subdir, dirs, files in tqdm(list(os.walk(sys.argv[1]))):
    for file in tqdm(files):
        if file.startswith('_'):
            no_underscore_file = file[1:]
        else:
            continue

        orig_file_path = os.path.join(subdir, file)
        file_path = os.path.join(subdir, no_underscore_file)
        if os.path.exists(file_path):
            os.remove(orig_file_path)
        else:
            os.rename(orig_file_path, file_path)
