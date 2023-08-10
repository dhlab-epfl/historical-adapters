import torch
import time
import math
import pickle
import pandas as pd

df = pd.read_csv('fr/HIPE-2022-v2.1-hipe2020-test-fr.tsv', sep='\t+')
df['org_index'] = df.index.tolist()
drop_df = df[df['NE-COARSE-LIT'].notna()]
drop_df = drop_df[drop_df['TOKEN'].notna()]

template_list = [" is a time entity.", " is a location entity.", " is a person entity.", " is an organization entity.",
                    " is an product entity.", " is not a named entity."]
entity_dict = {'time':0, 'loc':1, 'pers':2, 'org':3, 'prod':4, 'O':5}

def get_entities(seq, token_seq):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-PER']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3], ['PER', 4, 4]]
    """
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    tokens = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        type_ = chunk.split('-')[-1]

        if tag == 'O':
            chunks.append((tag, i, i))
            tokens.append((token_seq[i], 'O', i))

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
            try:
                tokens.append((' '.join(token_seq[begin_offset:i]), prev_type, begin_offset, i-1))
            except:
                print(token_seq[begin_offset:i])
                print(type(token_seq[begin_offset:i]))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return set(tokens), set(chunks)


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.
    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.
    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start

class InputExample():
    def __init__(self, words, labels, nat_labels):
        self.words = words
        self.labels = labels
        self.nat_labels = nat_labels

examples = []
words = []
labels = []
nat_labels = []

for i in range(len(drop_df)):
    line = drop_df.iloc[i]
    
# Every sentence
    if 'EndOfSentence' in line['MISC']:
        words.append(line['TOKEN'])
        labels.append(line['NE-COARSE-LIT'])
        
        if words:
            tokens, out = get_entities(labels, words+['O'])
            tokens = list(tokens)
            for t in range(len(tokens)):  
                tag = tokens[t][1]
                token = tokens[t][0]
                nat_labels.append(str(token) + template_list[entity_dict[tag]])
                try:
                    nat_labels.remove('O is not a named entity.')
                except:
                    pass
                    # print(nat_labels)
            example = InputExample(words=words, labels=labels, nat_labels=nat_labels)    
            
            examples.append(example)
            
            words = []
            labels = []
            nat_labels = []
    else:
        # splits = line.split(" ")
        words.append(line['TOKEN'])
        labels.append(line['NE-COARSE-LIT'])

# Last sentence
if words:
    tokens, out = get_entities(labels, words+['O'])
    tokens = list(tokens)
    for t in range(len(tokens)):  
        tag = tokens[t][1]
        token = tokens[t][0]
        nat_labels.append(str(token) + template_list[entity_dict[tag]])
        try:
            nat_labels.remove('O is not a named entity.')
        except:
            pass
    examples.append(InputExample(words=words, labels=labels, nat_labels=nat_labels))


#save it
print(len(examples))
# with open(f'parag-label-HIPE-test.pickle', 'wb') as file:
#     pickle.dump(examples, file)
#
# #load
# with open(f'parag-label-HIPE-test.pickle', 'rb') as file2:
#     examples = pickle.load(file2)
