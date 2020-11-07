import json

story_list = []
word_freq_file = open('out/stories_word_freq.json')
while True:
    story_path = word_freq_file.readline().strip()
    word_freq_str = word_freq_file.readline().strip()

    if story_path == '':
        break

    word_freq_dict = json.loads(word_freq_str)
    story_id = len(story_list)
    story_list.append(story_path)

    with open(f'../Stories/analyze_stats/word_freqs/{story_id}.txt', 'w') as out_file:
        for k, v in sorted(word_freq_dict.items(), key=lambda item: item[1], reverse=True):
            out_file.write(f'{k} {v}\n')
    break

with open('../Stories/analyze_stats/story_ids.txt', 'w') as out_file:
    for i, story_path in enumerate(story_list):
        out_file.write(f'{story_path}\n')