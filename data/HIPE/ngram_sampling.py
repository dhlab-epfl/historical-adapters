import pickle
import random
class InputExample():
    def __init__(self, words, labels, nat_labels):
        self.words = words
        self.labels = labels
        self.nat_labels = nat_labels # Ground truth

with open(f'parag-label-HIPE-test.pickle', 'rb') as file2:
    examples = pickle.load(file2)


def prediction(input_TXT):
    input_TXT_list = input_TXT

    entity_list = []
    for i in range(len(input_TXT_list)):
        words = []
        for j in range(1, min(9, len(input_TXT_list) - i + 1)):
            word = (' ').join(input_TXT_list[i:i+j])
            words.append(word)

        entity = template_entity(words, input_TXT, i) #[start_index,end_index,label,score]
        entity_list.append(entity)
    return entity_list

def template_entity(words, input_TXT, start):
    # input text -> template
    words_length = len(words)
    input_TXT = [input_TXT]*(6*words_length)

    template_list = [" is a time entity.", " is a location entity.", " is a person entity.", " is an organization entity.",
                    " is an product entity.", " is not a named entity."]
    temp_list = []
    for i in range(words_length):
        for j in range(len(template_list)):
            temp_list.append(words[i]+template_list[j])

    return temp_list

whole_entity_lst = []
for i in range(len(examples)):
    whole_entity_lst.append(prediction(examples[i].words))

total = []


for i in range(len(whole_entity_lst)):
    gt_lst = examples[i].nat_labels
    each = []
    all_no = []
    
    for j in range(len(whole_entity_lst[i])): # 각 sentence 의 모든 entity list

        for k in range(len(whole_entity_lst[i][j])):
        
            if whole_entity_lst[i][j][k] in gt_lst:
                temp = (whole_entity_lst[i][j][k], 'yes')
                each.append(temp) 
            else:
                if 'not a named entity' in whole_entity_lst[i][j][k]:
                    temp = (whole_entity_lst[i][j][k], 'yes')
                    each.appensd(temp) 
                else:
                    all_no.append((whole_entity_lst[i][j][k], 'no'))
                    # temp = (whole_entity_lst[i][j][k], 'no')   

            # each.append(temp) 
        sample_3 = random.sample(all_no, 3) 
        each.extend(sample_3)
    
    total.append(each)


class InputExample():
    def __init__(self, words, labels, nat_labels, all_entity):
        self.words = words
        self.labels = labels
        self.nat_labels = nat_labels # Ground truth
        self.all_entity = all_entity

whole = []
for i in range(len(examples)):
    whole.append(InputExample(words=examples[i].words, labels=examples[i].labels, nat_labels=examples[i].nat_labels, all_entity=total[i]))

with open(f'parag-label-sample3-ngram-HIPE-test.pickle', 'wb') as file:
    pickle.dump(whole, file) 
