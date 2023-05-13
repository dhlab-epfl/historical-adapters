import random

def create_one_example(question, context, choice, answer, test=False):
    input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"

    if test:
        output = "Answer: The answer is "
    else:
        output = f"Answer: The answer is {answer}."

    text = input + output
    text = text.replace("  ", " ").strip()

    return text


def get_question_text(problem):
    question = problem['question']
    return question

def get_context_text(problem):
    context = problem['hint']
    if context == "":
        context = "N/A"
    return context

def get_choice_text(probelm, options=['A', 'B', 'C', 'D', 'E']):
    choices = probelm['choices']
    choice_list = []
    for i, c in enumerate(choices):
        choice_list.append("({}) {}".format(options[i], c))
    choice_txt = " ".join(choice_list)
    return choice_txt

def get_answer(problem, options=['A', 'B', 'C', 'D', 'E']):
    return options[problem['answer']]


def build_prompt(example, test=False):

    question = get_question_text(example)
    context = get_context_text(example)
    choice = get_choice_text(example)
    answer = get_answer(example)

    prompt = create_one_example(question, context, choice, answer, test)

    return prompt


def get_pred_idx(prediction, choices, options=['A', 'B', 'C', 'D', 'E']):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return random.choice(range(len(choices)))