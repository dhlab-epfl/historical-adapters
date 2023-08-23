import json
from nltk.chunk import conlltags2tree
from nltk import pos_tag
from nltk.tree import Tree
import ast
import pickle
from nervaluate import Evaluator
import pprint

def get_entities(tokens, tags):
    tags = [tag.replace('S-', 'B-').replace('E-', 'I-') for tag in tags]
    pos_tags = [pos for token, pos in pos_tag(tokens)]

    conlltags = [(token, pos, tg)
                 for token, pos, tg in zip(tokens, pos_tags, tags)]
    ne_tree = conlltags2tree(conlltags)

    entities = []
    idx = 0
    char_position = 0  # This will hold the current character position

    for subtree in ne_tree:
        # skipping 'O' tags
        if isinstance(subtree, Tree):
            original_label = subtree.label()
            original_string = " ".join(
                [token for token, pos in subtree.leaves()])

            entity_start_position = char_position
            entity_end_position = entity_start_position + len(original_string)

            entities.append(
                (original_string,
                 original_label,
                 (idx,
                  idx + len(subtree)),
                    (entity_start_position,
                     entity_end_position)))
            idx += len(subtree)

            # Update the current character position
            # We add the length of the original string + 1 (for the space)
            char_position += len(original_string) + 1
        else:
            token, pos = subtree
            # If it's not a named entity, we still need to update the character
            # position
            char_position += len(token) + 1  # We add 1 for the space
            idx += 1

    return entities
def _read_conll(path, encoding='utf-8', sep=None, indexes=[0, 1], dropna=True):
    r"""
    Construct a generator to read conll items.
    :param path: file path
    :param encoding: file's encoding, default: utf-8
    :param sep: seperator
    :param indexes: conll object's column indexes that needed, if None, all columns are needed. default: None
    :param dropna: weather to ignore and drop invalid data,
            :if False, raise ValueError when reading invalid data. default: True
    :return: generator, every time yield (line number, conll item)
    """

    def parse_conll(sample):
        sample = list(map(list, zip(*sample)))
        # import pdb;
        # pdb.set_trace()
        sample = [sample[i] for i in indexes]
        for f in sample:
            if len(f) <= 0:
                raise ValueError('empty field')
        return sample

    with open(path, 'r', encoding=encoding) as f:
        sample = []
        start = next(f).strip()
        start = next(f).strip()

        data = []
        # if start != '':
        #     sample.append(
        #         start.split(sep)) if sep else sample.append(
        #         start.split())

        for line_idx, line in enumerate(f, 1):
            line = line.strip()

            if 'DOCSTART' in line:
                continue
            if '### ' in line:
                continue
            if "# id" in line:
                continue
            if "# " in line:
                continue
            if "Token" in line:
                continue
            if "TOKEN" in line:
                continue
            if 'hipe2022' in line:
                continue

            if line == '':
                if len(sample):
                    try:
                        print(sample)

                        res = parse_conll(sample)
                        sample = []
                        if ['TOKEN'] not in res:
                            if ['Token'] not in res:
                                data.append([line_idx, res])
                    except Exception as e:
                        if dropna:
                            print(
                                'Invalid instance which ends at line: {} has been dropped.'.format(line_idx))
                            sample = []
                            raise e
                            continue
                        raise ValueError(
                            'Invalid instance which ends at line: {}'.format(line_idx))
            elif 'EndOfSentence' in line:

                sample.append(
                    line.split(sep)) if sep else sample.append(
                    line.split())

                if len(sample):
                    try:
                        res = parse_conll(sample)
                        sample = []
                        if ['TOKEN'] not in res:
                            if ['Token'] not in res:
                                data.append([line_idx, res])
                    except Exception as e:
                        if dropna:
                            print(
                                'Invalid instance which ends at line: {} has been dropped.'.format(line_idx))
                            sample = []
                            raise e
                            continue
                        raise ValueError(
                            'Invalid instance which ends at line: {}'.format(line_idx))
            else:
                sample.append(
                    line.split(sep)) if sep else sample.append(
                    line.split())

        if len(sample) > 0:
            try:
                res = parse_conll(sample)
                if ['TOKEN'] not in res:
                    if ['Token'] not in res:
                        data.append([line_idx, res])
            except Exception as e:
                if dropna:
                    return
                print('Invalid instance ends at line: {}'.format(line_idx))
                raise e

        return data

def sentence_to_iob(sentence, entities):
    # Tokenize the sentence
    tokens = sentence

    # Initialize a list to store the labels
    labels = ["O"] * len(tokens.split())
    # print(entities)
    # Iterate over each entity in the entities list
    for entity in entities:
        # Split the entity text into tokens
        # if 'text' in entity:
        # print(entity)
        if 'text' in entity:
            entity_tokens = entity['text'].split()
            
            for i in range(len(tokens)):
                # Check if the sequence of tokens in the sentence matches the entity tokens
                if tokens[i:i+len(entity_tokens)] == entity_tokens:
                    # If it's a match, assign the appropriate labels
                    try:
                        labels[i] = "B-" + TAGS[entity['entity']]
                        for j in range(i+1, i+len(entity_tokens)):
                            labels[j] = "I-" + TAGS[entity['entity']]
                    except KeyError:
                        pass

        else:
            try:
                keys = list(entity.keys())
                vals = list(entity.values())
                # Case 1: {'UBS': 'PERSON', 'est': 'TIME', 'en': 'LOCATION', 'tant': 'PERSON', 'partie': 'ORGANIZATION', 'essentielle': 'PRODUCT'}
                    # key: token, val: tag
                if 'PERSON' or 'TIME' or 'LOCATION' or 'ORGANIZATION' or 'PRODUCT' in vals:
                
                    for i in range(len(keys)):
                        entity_tokens = keys[i].split()
                 
                        for j in range(len(tokens)):
                            if tokens[j:j+len(entity_tokens)] == entity_tokens:
                                try:
                                    labels[j] = "B-" + TAGS[vals[i]]
                                    for k in range(j+1, j+len(entity_tokens)):
                                        labels[k] = "I-" + TAGS[vals[i]]
                                except KeyError:
                                    pass
                                except TypeError:
                                    pass


                # Case 2: {'time': 'La course', 'location': 'avant son compatriote Wilfred Kigen à 25 ’ ’', 'person': 'Daid Mandago'}
                    # key: tag, val: token
                if 'person' or 'time' or 'location' or 'organization' or 'product' in keys:
                    for i in range(len(vals)):
                        entity_tokens = str(vals[i]).split()
                       
                        for j in range(len(tokens)):
                            if tokens[j:j+len(entity_tokens)] == entity_tokens:
                                try:
                                    labels[j] = "B-" + TAGS[keys[i].upper()]
                                    for k in range(j+1, j+len(entity_tokens)):
                                        labels[k] = "I-" + TAGS[keys[i].upper()]
                                except KeyError:
                                    pass
            

            
            except AttributeError:
                entity_tokens = list(entity)[0]
        
    return tokens, labels


if __name__ == '__main__':

    data_hipe = _read_conll('data/HIPE/HIPE-2022-v2.1-hipe2020-test-fr.tsv')
    print(len(data_hipe))
    data_hipe = [data[1] for data in data_hipe]

    true_tokens = [data[0] for data in data_hipe]
    true_tags = [data[1] for data in data_hipe]

    # print(true_tokens[1])
    import itertools

    true_tokens = list(itertools.chain(*true_tokens))
    true_tags = list(itertools.chain(*true_tags))
    print(len(true_tokens))

    import re
    filename = 'results/org-zeroshot-hipe-prompt1-result.json'
    # Replace 'filename.json' with your actual file path
    with open(filename, 'r') as f:
        data = json.load(f)

    output = data["output"]

    TAGS = {'LOCATION': 'loc',
            'ORGANIZATION': 'org',
            'PERSON': 'pers',
            'PRODUCT': 'prod',
            'TIME': 'time'}

    pred_tokens = []
    pred_tags = []
    predictions = []

    class InputExample():
        def __init__(self, words, labels, nat_labels):
            self.words = words
            self.labels = labels
            self.nat_labels = nat_labels

    #load it
    with open(f'./data/HIPE/parag-label-HIPE-test.pickle', 'rb') as file2:
        s1_new = pickle.load(file2)

    trues = []

    for i in range(len(s1_new)):
        trues.append(s1_new[i].labels)

    for item in output:
        # First, find the index where the OUTPUT part begins:
        output_index = item.index("OUTPUT:")

        sentence = item[item.index('INPUT:') + 6:output_index].strip()

        # tokens = ast.literal_eval(sentence)
        # pred_tokens.extend(tokens)
        pred_tokens.extend(sentence.split(' '))

        # Extract the output string (plus 7 for the length of the word "OUTPUT:")
        output_string = item[output_index + 7:].strip()
        output_string = output_string.replace("'", '"')
        # output_string = output_string.rstrip(", {").rstrip(", {") + "}]"

        # Find all dictionaries in the string
        matches = re.findall(r"\{[^}]*\}", output_string)
        # matches = re.findall(r"\([^}]*\)", output_string)

        # print(matches)
        # Parse each dictionary and store them in a list
        dictionaries = []
        print('-' * 100)
        # print(tokens)
        # print('*'*40)
        print(matches)
        for match in matches:
            try:
                # Replace problematic characters
                # match = match.replace(" '", "'").replace(" '", "'")
                
                match = match.replace('d "', "d '")
                match = match.replace('s "', "s '")
                match = match.replace('S "', "S '")
                match = match.replace('l "', "l '")
                match = match.replace('L "', "L '")
                match = match.replace('o " n', "o ' n")
                match = match.replace('M . "', "M . '")
                match = match.replace("d '}", 'd "}')
                match = match.replace('1 " j', "1 ' j")
                match = match.replace('n " e', "n ' e")
                match = match.replace('D " E', "D ' E")
                match = match.replace("es '", 'es "')
                match = match.replace('n " i', "n ' i")
                match = match.replace('r " e', "r ' e")
                match = match.replace('u " a', "u ' a")
                match = match.replace('e " v', "e ' v")
                match = match.replace('> }', '>" }')
                match = match.replace('s" o', "s' o")
                match = match.replace('C " e', "C ' e")
                match = match.replace('s " Q', "s ' Q")
                match = match.replace("s '}", 's "}')
                match = match.replace('n " é', "n ' é")
                match = match.replace('s" e', "s' e")
                match = match.replace('u " u', "u ' u")

                dictionary = ast.literal_eval(match)
                dictionaries.append(dictionary)
                
                # list of dictionary for each sentence
                # print(dictionaries)

                # {token:tag}


                # print(dictionary)
                # if dictionary['text'] not in sentence:
                #     print(match, '------>', sentence)
            except SyntaxError:
                # Print the problematic string for inspection and skip
                # print(f"Skipping problematic string: {match}")
                continue

        tokens, tags = sentence_to_iob(sentence, dictionaries)
        # print(tokens, tags)
        pred_tags += tags
        predictions.append(tags)



    print(len(true_tags), len(pred_tags), len(true_tokens), len(pred_tokens))
    from conlleval import evaluate
    evaluate(true_tags, pred_tags, verbose=True)

    # evaluator = Evaluator(trues, predictions, tags=['loc', 'pers', 'prod', 'time', 'org'], loader="list")

    # results, results_by_tag = evaluator.evaluate()

    # pprint.pprint(results)
    # print()
    # print()
    # pprint.pprint(results_by_tag)
