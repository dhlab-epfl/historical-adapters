import pandas as pd
import numpy as np
import json

def random_choice_except(a: int, excluding: int, size=None, replace=True):
    # generate random values in the range [0, a-1)
    choices = np.random.choice(a-1, size, replace=replace)
    # shift values to avoid the excluded number
    return choices + (choices >= excluding)

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
            tokens.append((token_seq[i], 'O'))

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
            tokens.append((' '.join(token_seq[begin_offset:i]), prev_type))
            
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


if __name__ == "__main__":

    df = pd.read_csv('../fr/HIPE-2022-v2.1-hipe2020-train-fr.tsv', sep='\t+')
    df['org_index'] = df.index.tolist()
    drop_df = df[df['NE-COARSE-LIT'].notna()]
    drop_df = drop_df[drop_df['TOKEN'].notna()]

    seq = drop_df['NE-COARSE-LIT'].tolist()
    token_seq = drop_df['TOKEN'].tolist() + ['O']


    token_outputs, outputs = get_entities(seq, token_seq)

    template_list = [" is a time entity.", " is a location entity.", " is a person entity.", " is an organization entity.",
                        " is an product entity.", " is not a named entity."]
    entity_dict = {'time':0, 'loc':1, 'pers':2, 'org':3, 'prod':4, 'O':5}
    inv_entity_dict = {v: k for k, v in entity_dict.items()}

    total = {}
    pos = [] # ground truth
    false_pos = [] # another type of tag 
    # non = [] # another token with ground truth tag
    null = [] # not entity
    token_outputs = list(token_outputs)

    for i in range(len(token_outputs)):
        tag = token_outputs[i][1]
        token = token_outputs[i][0]
        if tag == 'O':
            null.append(str(token) + template_list[entity_dict[tag]])
        else:
            pos.append(str(token) + template_list[entity_dict[tag]])
            rand_idx = random_choice_except(6, entity_dict[tag])
            rand_tag = inv_entity_dict[rand_idx]
            false_pos.append(str(token) + template_list[entity_dict[rand_tag]])
        
    total['positive'] = pos
    total['flase_pos'] = false_pos
    total['null'] = null

    with open('HIPE_converted_train_fr.json', 'w', encoding='utf8') as fp:     
        json.dump(total, fp, indent=4, ensure_ascii=False)
