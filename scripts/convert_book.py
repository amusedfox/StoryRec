import sys

sys.path.append('.')
from story.base import BaseStory
import os

for filename in os.listdir('../books'):
    file_path = os.path.join('../books', filename)

    info = filename.split('-')
    if len(info) != 2:
        raise ValueError('There is a "-" in the book title/author')

    with open(file_path) as in_file:
        text = in_file.read()

    base = BaseStory()
    base.content = text
    base.author = info[0]
    base.title = info[1]
    base.word_count = len(text.split())
    base.write()