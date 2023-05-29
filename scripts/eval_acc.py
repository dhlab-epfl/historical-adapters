import os
import json
import argparse
import warnings
import pandas as pd
from datasets import load_dataset

warnings.filterwarnings('ignore')


def get_acc_with_contion(res_pd, key, values):
    if isinstance(values, list):
        total_pd = res_pd[res_pd[key].isin(values)]
    else:
        total_pd = res_pd[res_pd[key] == values]
    correct_pd = total_pd[total_pd['true_false'] == True]
    if len(total_pd) == 0:
        return -1
    acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100)
    return acc


def get_scores(result_file, data_file):
    # read result file
    results = json.load(open(result_file))["prediction"]
    num = len(results)
    assert num == 4241
    #print("number of questions:", num)

    # construct pandas data
    res_pd = pd.DataFrame(data_file).T

    # update data
    for index, row in res_pd.iterrows():

        label = row['answer']
        pred = int(results[index])
        res_pd.loc[index, 'pred'] = pred
        res_pd.loc[index, 'true_false'] = (label == pred)

    # accuracy scores
    acc_average = len(res_pd[res_pd['true_false'] == True]) / num * 100
    #assert result_file.split('_')[-1] == "{:.3f}.json".format(acc_average)

    scores = {
        'acc_natural':
        get_acc_with_contion(res_pd, 'subject', 'natural science'),
        'acc_social':
        get_acc_with_contion(res_pd, 'subject', 'social science'),
        'acc_language':
        get_acc_with_contion(res_pd, 'subject', 'language science'),
        'acc_grade_1_6':
        get_acc_with_contion(res_pd, 'grade', ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']),
        'acc_grade_7_12':
        get_acc_with_contion(res_pd, 'grade', ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']),
        'acc_average':
        "{:.2f}".format(acc_average),
    }
    topics = ['punctuation', 'literacy-in-science', 'verbs', 'pronouns', 'civics', 'culture', 'word-study', 'economics', 'physics', 'units-and-measurement', 'science-and-engineering-practices', 'reading-comprehension', 'global-studies', 'grammar', 'figurative-language', 'us-history', 'writing-strategies', 'world-history', 'reference-skills', 'biology', 'earth-science', 'phonological-awareness', 'capitalization', 'chemistry', 'vocabulary', 'geography']
    for t in topics:
        scores['acc_' + t] = get_acc_with_contion(res_pd, 'topic', t)

    return scores


def print_scores(scores):
    latex_output = ""
    for key, score in scores.items():
        print(f"{key[4:]}: \t{score}")
        latex_output += f"& {score} "
    latex_output += "\\\\"
    print(latex_output)


if __name__ == '__main__':

    dataset = load_dataset("derek-thomas/ScienceQA")
    testset = dataset['test']

    scores = get_scores('./results/org-finetuned-result-latest.json', testset)

    for i in scores:
        print_scores(i)
