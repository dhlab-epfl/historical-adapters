import json
from nltk.chunk import conlltags2tree
from nltk import pos_tag
from nltk.tree import Tree
import ast
from nervaluate import Evaluator
import pickle 
import pprint


if __name__ == '__main__':

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

    import re
    filename = 'results/org-finetuned-hipe-prompt2-result.json'
    # Replace 'filename.json' with your actual file path
    with open(filename, 'r') as f:
        data = json.load(f)

    output = data["output"]

    predictions = []
    
    for i in range(len(output)):
        try:
            sentence = output[i].split('Input: ')[1].split('\n\n\n        Output:')[0]
        except:
            sentence = '[]'
            
        tokens = ast.literal_eval(sentence)
        
        try:
            output_string = output[i].split('Output: \n        ')[1]
        except:
            output_string = '[]'
        # output_string = output_string.replace("'", '"')
   
        # Find all dictionaries in the string
        if output_string[-1] == ']':
            matches = output_string
 
        dictionaries = []

        matches = matches.replace('d "', "d '")
        matches = matches.replace('s "', "s '")
        matches = matches.replace('S "', "S '")
        matches = matches.replace('l "', "l '")
        matches = matches.replace('L "', "L '")
        matches = matches.replace('o " n', "o ' n")
        matches = matches.replace('M . "', "M . '")
        matches = matches.replace("d '}", 'd "}')
        matches = matches.replace('1 " j', "1 ' j")
        matches = matches.replace('n " e', "n ' e")
        matches = matches.replace('D " E', "D ' E")
        matches = matches.replace("es '", 'es "')
        matches = matches.replace('n " i', "n ' i")
        matches = matches.replace('r " e', "r ' e")
        matches = matches.replace('u " a', "u ' a")
        matches = matches.replace('e " v', "e ' v")
        matches = matches.replace('> }', '>" }')
        matches = matches.replace('s" o', "s' o")
        matches = matches.replace('C " e', "C ' e")
        matches = matches.replace('s " Q', "s ' Q")
        matches = matches.replace("s '}", 's "}')
        matches = matches.replace('n " é', "n ' é")
        matches = matches.replace('s" e', "s' e")
        matches = matches.replace('u " u', "u ' u")
        matches = matches.replace("(’’", "('’’'")
        matches = matches.replace("(’'", "('’'")
        matches = matches.strip()

       
        try:
            dictionary = eval(matches)
         
        except SyntaxError:
            print(matches)
            pass

        
        try:
            new_dictionary = [ele[1][:2] + ele[1].split('-')[1].lower() if ele[1] != 'O' else 'O' for ele in dictionary]
            new_dictionary += ['O'] * (len(tokens) - len(new_dictionary))
            dictionaries += new_dictionary    
        
        # for the case only one token in dictionary e.g., ('.', 'O')
        except IndexError:
            dictionaries.append(dictionary[1])

        predictions.append(dictionaries)

    
    evaluator = Evaluator(trues, predictions, tags=['loc', 'pers', 'prod', 'time', 'org'], loader="list")

    results, results_by_tag = evaluator.evaluate()

    pprint.pprint(results)
    print()
    print()
    pprint.pprint(results_by_tag)
