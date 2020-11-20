#!/bin/bash

python3 tag_assignment/story_tfidf.py "$STORY_REC"/story_txt/ "$STORY_REC"/lemm_list "$STORY_REC"/ngram_freq "$STORY_REC"/story_tfidf "$STORY_REC"/assets/ngram_doc_freq.txt -s True
if [ $? -ne 0 ]; then
    exit 1
fi

python3 tag_assignment/tag_tfidf.py "$STORY_REC"/story_html/ "$STORY_REC"/story_tfidf "$STORY_REC"/tag_stories "$STORY_REC"/tag_stats
if [ $? -ne 0 ]; then
    exit 1
fi

python tag_assignment/weight_predict.py "$STORY_REC"/story_html "$STORY_REC"/story_tfidf "$STORY_REC"/tag_stats "$STORY_REC"/tag_stories "$STORY_REC"/save_vectors "$STORY_REC"/results
if [ $? -ne 0 ]; then
    exit 1
fi
