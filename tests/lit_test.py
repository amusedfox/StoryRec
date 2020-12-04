import pytest
from story.lit import *


@pytest.mark.parametrize("url, expected_result", [
    ('https://www.literotica.com/s/the-infinity-device', 2),
    ('https://www.literotica.com/s/i-bet-youll-like-it', 4),
    ('https://www.literotica.com/s/a-bisexual-somali-love-story-ch-02', 1),
])
def test_get_page_count(url, expected_result):
    n_pages, _ = get_page_count(url, '')
    assert n_pages == expected_result, n_pages
