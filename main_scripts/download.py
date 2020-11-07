from typing import NewType, List
import requests
from bs4 import BeautifulSoup
import os
import sys
from story.ff import FFStory
from story.lit import LitStory

WARNING_COLOR = '\033[93m'
ERROR_COLOR = '\033[91m'
NORMAL_COLOR = '\033[0m'
GREEN_COLOR = '\033[92m'

# Lit workflow
# Get list of categories
# For category:
#   For story_link in category:
#       Download story

# ff workflow
# Get main_page
# For story_link in main_page:
#   Download story


story_type = {
    'lit': LitStory,
    'ff': FFStory
}


def main():
    if len(sys.argv) < 2:
        print(f'Needs 1 argument: story_id_file')
        return

    with open(sys.argv[1]) as in_file:
        for line in in_file:
            temp = line.split('/')
            s_index = temp.index('s')
            story_id = temp[s_index + 1]
            FFStory(story_dir='../fanfic', story_id=story_id)


if __name__ == '__main__':
    main()
