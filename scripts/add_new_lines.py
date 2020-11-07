with open('out/stories_word_freq.json') as in_file, open('temp/fixed.json', 'w') as out_file:
    while True:
        text = in_file.read(1000000)
        if text == '':
            break

        text = text.replace('"\n', '\n')
        text = text.replace('\n"', '\n')
        out_file.write(text)
