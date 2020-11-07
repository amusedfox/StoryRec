import os

removed_count = 0

with open('out/lit_short_story_list.txt') as in_file:
    for line in in_file:
        file_path = line.strip()
        try:
            os.remove(os.path.join(f'../Stories', file_path))
        except FileNotFoundError:
            pass

        removed_count += 1

print(f"Removed {removed_count} stories")
