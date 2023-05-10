import pandas as pd
import json
import os
import argparse
import random


def csv_to_json(csv_file, val_ids, test_ids, train_ids, idx):
    # Read the CSV file using pandas
    df = pd.read_csv(csv_file)

    # Initialize an empty dictionary for the final JSON data
    json_data = {}

    # Iterate through each row of the DataFrame and create the JSON structure
    for index, row in df.iterrows():
        if "test" in str(row['id']):
            test_ids.append(str(idx))
        elif "val" in str(row['id']):
            val_ids.append(str(idx))
        else:
            train_ids.append(str(idx))

        json_data[str(idx)] = {
            "question": row['question'],
            "choices": [row['org_answer']],
            "answer": 0, # Assuming the correct answer is always the second choice
            "hint": "",
            "image": "",
            "task": "open choice",
            "grade": "", # Add grade information if available
            "subject": "", # Add subject information if available
            "topic": row['source'], # Add topic information if available
            "category": "", # Add category information if available
            "skill": "", # Add skill information if available
            "lecture": "", # Add lecture information if available
            "solution": "",
            "split": "test" if row['id'].startswith("test") else "train"
        }

        idx += 1

    return json_data, idx


def process_csv_files(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    val_ids, test_ids, train_ids = [], [], []
    all_problems = {}
    idx = 0
    # Process each file in the folder
    for file in files:
        # Check if the file is a CSV file
        if file.endswith('.csv'):
            csv_file = os.path.join(folder_path, file)
            json_data, idx = csv_to_json(csv_file, val_ids, test_ids, train_ids, idx)

            # Merge the JSON data into the all_problems dictionary
            all_problems.update(json_data)

            # # Save the JSON data to a file
            # output_file = os.path.join(folder_path, f"{os.path.splitext(file)[0]}.json")
            # with open(output_file, 'w') as f:
            #     f.write(json_data)

    # Save the combined JSON data to a file called problems.json
    problems_json = json.dumps(all_problems, indent=2)
    problems_file = os.path.join(folder_path, 'problems.json')
    with open(problems_file, 'w') as f:
        f.write(problems_json)

    # Generate pid_splits.json
    pid_splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }
    pid_splits_json = json.dumps(pid_splits, indent=2)

    # Save pid_splits.json to the folder
    pid_splits_file = os.path.join(folder_path, 'pid_splits.json')
    with open(pid_splits_file, 'w') as f:
        f.write(pid_splits_json)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data/ArchivalQA/data')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)
    folder_path = args.data_root
    process_csv_files(folder_path)
