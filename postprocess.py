import json


def ngram_to_iob(json_file, token_df):
    with open(json_file, 'r') as file:
        data = json.load(file)
        outputs = data['output']

    labels = {token: 'O' for token in token_df['Token']}

    for output in outputs:
        question = output.split('\n')[1]
        answer = output.split('\n')[3].split(':')[1].strip()

        if answer == 'yes':
            if ' is a ' in question:
                entity_type = question.split(' is a ')[1].split(' entity.')[0]
                ngram = question.split(' is a ')[0]
            elif ' is an ' in question:
                entity_type = question.split(' is an ')[1].split(' entity.')[0]
                ngram = question.split(' is an ')[0]

            tokens = ngram.split(' ')
            if 'is not a named entity' in question:
                for token in tokens:
                    labels[token] = 'O'
            else:
                labels[tokens[0]] = 'B-' + entity_type
                for token in tokens[1:]:
                    labels[token] = 'I-' + entity_type

    token_df['Predicted'] = token_df['Token'].map(labels)
    return token_df


df = ngram_to_iob('data/org-finetuned-hipe-result-ngram-sample.json')
print(df)
