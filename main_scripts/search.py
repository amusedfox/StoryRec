from bs4 import BeautifulSoup
import os
import sys
import pathlib
import re

good_phrases = []
with open('../test_set/lit/good_phrases.txt') as in_file:
    for line in in_file:
        good_phrases.append(line.strip())

bad_phrases = []
with open('../test_set/lit/bad_phrases.txt') as in_file:
    for line in in_file:
        bad_phrases.append(line.strip())

print(good_phrases)
print(bad_phrases)
if not os.path.isdir(f'../../stories/{sys.argv[1]}'):
    print('Not a valid directory')
    sys.exit()

found_list = []
for subdir, dirs, files in os.walk(f'../../stories/{sys.argv[1]}'):
    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith('.html'):
            with open(filepath) as in_file:
                story = in_file.read().lower()
                phrase_count = sum(map(story.count, good_phrases))
                phrase_count -= sum(map(story.count, bad_phrases))

                if phrase_count > 0:
                    found_list.append((phrase_count, filepath))
                    print(phrase_count, filepath)

with open(f'test_set/found_list_{sys.argv[1]}.txt', 'w') as out_file:
    found_list.sort(key=lambda x: x[0], reverse=True)
    for found in found_list:
        out_file.write(f'{found[0]}\t{found[1]}\n')
