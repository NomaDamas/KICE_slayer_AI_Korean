import json
import os
from typing import List

from dotenv import load_dotenv
import openai
from prompts import basic_prompt, talk_prompt, literature_prompt
import click
from tqdm import tqdm
import time
import logging


def load_test(filepath: str):
    # check if file exists
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f'File not found: {filepath}')

    with open(filepath, 'rb') as f:
        test = json.load(f)
    total_score_test(test)
    return test


def total_score_test(data):
    total_score = 0
    for pa in data:
        for problem in pa["problems"]:
            total_score += problem["score"]
    assert (total_score == 100)
    print("test passed")


def set_openai_key():
    load_dotenv()
    openai.api_key = os.environ["OPENAI_API_KEY"]


def get_answer_one_problem(data, paragraph_num: int, problem_num: int, prompt_func: callable = basic_prompt):
    problem = data[paragraph_num]["problems"][problem_num]
    no_paragraph = False
    if "no_paragraph" in list(problem.keys()):
        no_paragraph = True
    if "question_plus" in list(problem.keys()):
        question_plus_text = problem["question_plus"]
    else:
        question_plus_text = ""
    return prompt_func(paragraph=data[paragraph_num]["paragraph"],
                                             question=problem["question"],
                                             choices=problem["choices"],
                                             question_plus=question_plus_text,
                                             no_paragraph=no_paragraph)


def get_prompt_by_type(type_num: int) -> callable:
    # 0 : 비문학, 1 : 문학, 2 : 화법과 작문, 3 : 문법
    if type_num == 0:
        return talk_prompt
    elif type_num == 1:
        return literature_prompt
    elif type_num == 2:
        return talk_prompt
    else:
        print("type_num must be 0, 1, 2. 문법 is not supported yet")
        return basic_prompt


def save_results_txt(data, save_path: str, answer_list: List[str]):
    solutions = list()
    for pa in data:
        for problem in pa["problems"]:
            solutions.append(problem["answer"])

    scores = list()
    for pa in data:
        for problem in pa["problems"]:
            scores.append(problem["score"])

    f = open(save_path, 'w', encoding='UTF-8')
    for i, item in enumerate(answer_list):
        txt = f'{i + 1}번 문제 : {item}\n정답 : {solutions[i]}\n배점 : {scores[i]}\n----------------------------\n'
        print(txt)
        f.write(txt)
    f.close()
    print("saved DONE")


@click.command()
@click.option('--test_file', help='test file path')
@click.option('--save_path', help='save path')
def main(test_file, save_path):
    set_openai_key()
    test = load_test(test_file)
    answer_list = list()
    for paragraph_index, paragraph in enumerate(test):
        prompt_func = get_prompt_by_type(int(paragraph["type"]))
        for problem_index, problem in tqdm(enumerate(paragraph["problems"])):
            answer = get_answer_one_problem(test, paragraph_index, problem_index, prompt_func)
            logging.basicConfig(filename=save_path.split(".")[0] + "_log.log", level=logging.INFO)
            logging.info(answer)
            answer_list.append(answer)
            time.sleep(20)
    save_results_txt(test, save_path, answer_list)


if __name__ == "__main__":
    main()
