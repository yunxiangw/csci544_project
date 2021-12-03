#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import numpy as np
from tqdm import tqdm

sys.path.append('../..')
from dataset.emotional.emotional import judge_emotional
from dataset.event.event import judge_event
sys.path.remove('../..')


def judge(text):
    emo = judge_emotional(text)
    eve = judge_event(text)
    if (emo < 0.7 and eve < 0.7) or abs(emo - eve) < 0.3:
        # normal type, no specific style
        return 2, emo, eve, 0
    return np.argmax([emo, eve]), emo, eve, 0


if __name__ == '__main__':
    import pandas as pd

    path = sys.argv[1]
    data = pd.read_csv(path)
    example_num = len(data)
    sent_keys = [f'sentence{i+1}' for i in range(5)]
    for idx in tqdm(range(example_num)):
        story = []
        for key in sent_keys:
            story.append(data[key][idx].replace('.', ' .').replace(',', ' ,'))

        res = judge(' '.join(story[1:]))

        data.loc[idx, 'label'] = res[0]
        if pd.isnull(data.loc[idx, 'emotional']):
            data.loc[idx, 'normal'] = data.loc[idx, 'event']
        elif pd.isnull(data.loc[idx, 'event']):
            data.loc[idx, 'normal'] = data.loc[idx, 'emotional']
        else:
            data.loc[idx, 'normal'] = data.loc[idx, 'event'] + data.loc[idx, 'emotional']
    data.to_csv('../GLUCOSE_final.csv', index=False)

