import pytest
import sys
import os

sys.path.append('.')
from story.lit import *

@pytest.mark.parametrize("url, expected_result", [
    ('https://www.literotica.com/s/the-infinity-device', 2),
    ('https://www.literotica.com/s/i-bet-youll-like-it', 4),
    ('https://www.literotica.com/s/a-bisexual-somali-love-story-ch-02', 1),
])
def test_get_page_count(url, expected_result):
    n_pages, _ = get_page_count(url, '')
    assert n_pages == expected_result, n_pages


@pytest.mark.parametrize("url, expected_result", [
    ('https://www.literotica.com/s/a-class-in-psychology',
     'tests/test_data/AmberLion - A Class In Psychology'),
    ('https://www.literotica.com/s/90-seconds-of-lust',
     'tests/test_data/androgyne30 - 90 Seconds of Lust'),
    ('https://www.literotica.com/s/a-bets-a-bet-3',
     "tests/test_data/Anonoauthor - A Bet's a Bet"),
])
def test_lit_download(url, expected_result):

    try:
        os.removedirs('temp')
    except OSError:
        pass

    story = LitStory(story_url=url, save_html_dir='temp', save_txt_dir='temp')
    story.add_tag('anal')
    output, content = story.save(write_to_disk=False)

    with open(expected_result + '.html') as in_file:
        expected_output = in_file.read()

    with open(expected_result + '.txt') as in_file:
        expected_content = in_file.read()

    assert output == expected_output
    assert content == expected_content
