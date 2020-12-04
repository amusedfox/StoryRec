import sys
import pandas as pd
import pytest

sys.path.append('.')
from story_stats.stat_func import *
from story_stats.story_stats import *
from story_stats.util import syllables
from story.base import BaseStory

# pytest -s (to see normal print statements)

@pytest.mark.parametrize("text, expected_result", [
    ('He said, "Why are you so rude?"', 20/31),
    ('Why are you so rude?', 0),
    ('"Why are you so rude?', 0),
    ('Why are you so rude?"', 0),
    ('"Why are you so rude?"', 10/11),
])
def test_quote_density(text, expected_result):
    base = BaseStory()
    base.content = text
    quote_density = get_quote_density(base)
    assert pytest.approx(quote_density) == expected_result, f'{quote_density} != {expected_result}'


@pytest.mark.parametrize("text, expected_result", [
    ('He said, "Why are you so rude?" I replied, "I hate you."', (31 + 24) / 2)
])
def test_avg_sent_len(text, expected_result):
    base = BaseStory()
    base.content = text
    avg_sent_len = get_avg_sent_len(base)
    assert pytest.approx(avg_sent_len) == expected_result, f'{avg_sent_len} != {expected_result}'

@pytest.mark.parametrize("text, expected_result", [
    ('hi', 1),
    ('actual', 3),
    ('winter', 2),
    ('ravioli', 4),
    ('playful', 2),
    ('lovey', 2),
    ('Siberia', 4),
    ('karate', 3),
    ('readier', 3)
])
def test_syllables(text, expected_result):
    # base = BaseStory()
    # base.content = text
    # avg_sent_len = get_avg_sent_len(base)
    # assert pytest.approx(avg_sent_len) == expected_result, f'{avg_sent_len} != {expected_result}'

    assert syllables(text) == expected_result


class TestStoryStats:
    def test_get_stories_stats(self):
        parent_dir = 'test_set/test_stories/pytest'
        output_file_path = 'out/pytest_story_values.csv'
        df = get_stories_stats(parent_dir, output_file_path=output_file_path)

        assert len(df.index) == 1, f'{len(df.index)} rows were found: {list(df.index)}'
        assert df.index.values[0] == 'test_category/test_author - test_title.html', f'{df.index.values[0]}'

        assert len(df.columns) == len(STAT_NAME_FUNCS + META_INFO_LIST), f'{len(df.columns)} columns were found: {list(df.columns)}'

        row = df.iloc[0]

        assert row['lex_density'] == 0.266
        assert pytest.approx(row['avg_sent_len']) == 74.219858
        assert pytest.approx(row['avg_word_len']) == 4.056886
        assert pytest.approx(row['quote_density'], abs=1e-6) == 0.017319
        assert pytest.approx(row['commas_per_word'], abs=1e-6) == 0.061

    def test_remove_stat_from_file(self):
        remove_stat_from_file('lex_density', 'out/pytest_story_values.csv')
        df = pd.read_csv('out/pytest_story_values.csv', index_col=0)

        assert len(df.index) == 1, f'{len(df.index)} rows were found: {list(df.index)}'
        assert df.index.values[0] == 'test_category/test_author - test_title.html', f'{df.index.values[0]}'

        assert len(df.columns) == len(STAT_NAME_FUNCS + META_INFO_LIST) - 1, f'{len(df.columns)} columns were found: {list(df.columns)}'

        assert 'lex_density' not in df.columns, 'lex_density is still in df columns'

        row = df.iloc[0]

        assert pytest.approx(row['avg_sent_len']) == 74.219858
        assert pytest.approx(row['avg_word_len']) == 4.056886
        assert pytest.approx(row['quote_density'], abs=1e-6) == 0.017319
        assert pytest.approx(row['commas_per_word'], abs=1e-6) == 0.061

    def test_get_stories_stats_with_update(self):
        parent_dir = 'test_set/test_stories/pytest'
        output_file_path = 'out/pytest_story_values.csv'
        df = get_stories_stats(parent_dir, output_file_path=output_file_path,
                               stories_stats_pd_file_path=output_file_path,
                               update_col=True)
        os.remove(output_file_path)

        assert len(df.index) == 1, f'{len(df.index)} rows were found: {list(df.index)}'
        assert df.index.values[0] == 'test_category/test_author - test_title.html', f'{df.index.values[0]}'

        assert len(df.columns) == len(STAT_NAME_FUNCS + META_INFO_LIST), f'{len(df.columns)} columns were found: {list(df.columns)}'

        row = df.iloc[0]

        assert row['lex_density'] == 0.266
        assert pytest.approx(row['avg_sent_len']) == 74.219858
        assert pytest.approx(row['avg_word_len']) == 4.056886
        assert pytest.approx(row['quote_density'], abs=1e-6) == 0.017319
        assert pytest.approx(row['commas_per_word'], abs=1e-6) == 0.061
        
        
def test_remove_punct():
    test_str = 'asdfasjdfi   sdfi * #7 @()  @&#)@_ @ _@ #$(–\n'
    test_str = remove_punct(test_str)
    assert test_str == 'asdfasjdfi   sdfi        \n', f'|{test_str}|'


def test_remove_outliers():
    num_list = [
        [1, 1.2, 10],
        [8, 1.1, 11],
        [1.2, 1.15, 9],
        [1.1, 1.14, 30],
        [0.9, 1.13, 9.5],
        [0.95, 1.12, 9.9],
        [0.89, 1.09, 9.8],
        [0.899, 1.091, 10.5],
        [0.91, 1.21, 10.6],
        [0.923, 1.212, 10.64],
        [0.942, 1.211, 10.63],
        [0.999, 1.213, 10.62],
    ]

    correct_num_list = [
        [1, 1.2, 10],
        [1.2, 1.15, 9],
        [0.9, 1.13, 9.5],
        [0.95, 1.12, 9.9],
        [0.89, 1.09, 9.8],
        [0.899, 1.091, 10.5],
        [0.91, 1.21, 10.6],
        [0.923, 1.212, 10.64],
        [0.942, 1.211, 10.63],
        [0.999, 1.213, 10.62],
    ]

    df = pd.DataFrame(num_list)
    correct_df = pd.DataFrame(correct_num_list)

    remove_outliers(df=df)
    df.reset_index(drop=True, inplace=True)

    assert df.equals(correct_df), f'\n{df}\n\n{correct_df}'
