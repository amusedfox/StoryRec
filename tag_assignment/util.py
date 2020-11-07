import os


def search_dir(dir_to_search, file_ext: str, n_files_to_use=-1):
    """Returns list of file paths under directory"""

    assert file_ext[0] == '.', f'File extension: {file_ext} does not have a ' \
                               f'period'

    file_count = 0
    file_path_list = []
    for root, dirs, files in os.walk(dir_to_search):
        for file_name in files:
            if not file_name.endswith(file_ext):
                continue

            file_path_list.append(os.path.join(root, file_name))

            file_count += 1
            if file_count == n_files_to_use:
                return file_path_list

    return file_path_list


def load_dir(dir_to_search, file_ext: str, n_files_to_use=-1):
    """Generator function that yields (Relative file path, file contents)"""

    assert file_ext[0] == '.', f'File extension: {file_ext} does not have a ' \
                               f'period'

    file_count = 0
    for root, dirs, files in os.walk(dir_to_search):
        for file_name in files:
            if not file_name.endswith(file_ext):
                continue

            rel_file_path = os.path.join(root.rsplit('/', 1)[-1], file_name)
            file_path = os.path.join(root, file_name)
            with open(file_path) as in_file:
                yield rel_file_path, in_file.read()

            file_count += 1
            if file_count == n_files_to_use:
                return


def dict_to_file(file_path, dict_to_write, descending=True):
    with open(file_path, 'w') as out_file:
        for key, value in sorted(dict_to_write.items(),
                                 key=lambda l: l[1], reverse=descending):
            out_file.write(f'{key} {value}\n')

