import json
import os
from typing import List

import click
import openai
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms import VLLMOpenAI
from langchain.schema import StrOutputParser
from tqdm import tqdm

from prompts import literature_prompt, grammar_prompt, basic_prompt_plus, basic_prompt, active_prompt_plus, \
    active_prompt


def load_test(filepath: str):
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
        return literature_prompt
    elif type_num == 1:
        return literature_prompt
    elif type_num == 2:
        return literature_prompt
    else:
        return grammar_prompt


def cost_calc(model: str, input_token: int, output_token: int) -> float:
    if model == "gpt-4-1106-preview":
        return input_token * 0.00001 + output_token * 0.00003
    elif model == "gpt-4":
        return input_token * 0.00003 + output_token * 0.00006
    elif model == "gpt-3.5-turbo-1106":
        return input_token * 0.000001 + output_token * 0.000002
    elif model == "gpt-3.5":
        return input_token * 0.0000015 + output_token * 0.000002
    elif model == "HCX-003":
        return input_token * 0.005 + output_token * 0.005


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


def save_result_pd(save_path: str, answer_list):
    if not save_path.endswith('.csv'):
        raise ValueError('save_path must be a csv file')
    df = pd.DataFrame(answer_list, columns=['id', 'problem_num', 'gt_answer', 'pred', 'score'])
    df.to_csv(save_path, index=False, encoding='utf-8-sig')


def select_model(model_name: str):
    if model_name == 'gpt-4':
        return ChatOpenAI(model_name='gpt-4')
    elif model_name == 'synatra':
        return VLLMOpenAI(model_name="maywell/Synatra-7B-v0.3-base", openai_api_base=os.getenv('VLLM_BASE_URL'),
                          openai_api_key="")
    elif model_name == 'gpt-3':
        return ChatOpenAI(model_name='gpt-3.5-turbo-16k')
    else:
        raise ValueError('model_name must be one of gpt-4, gpt-3, synatra')


def main_func(test_file, save_path, model_name, start_num=0, end_num=50, run_list=None,
              prompt_base=basic_prompt, prompt_plus=basic_prompt_plus):
    load_dotenv()
    set_openai_key()
    test = load_test(test_file)
    answer_list = list()
    model = select_model(model_name)
    i = 0
    if run_list is None:
        run_list = list(range(1, 46))
    for paragraph_index, paragraph in enumerate(test):
        for problem_index, problem in tqdm(enumerate(paragraph["problems"])):
            i += 1
            if i < start_num:
                continue
            if i > end_num:
                break
            if i not in run_list:
                continue

            if "no_paragraph" in list(problem.keys()):
                paragraph_text = ""
            else:
                paragraph_text = paragraph['paragraph']
            if "question_plus" in list(problem.keys()):
                question_plus_text = problem["question_plus"]
                prompt = prompt_plus
                runnable = prompt | model | StrOutputParser()
                answer = runnable.invoke({
                    "question": problem["question"],
                    "paragraph": paragraph_text,
                    "question_plus": question_plus_text,
                    "choices_1": problem["choices"][0],
                    "choices_2": problem["choices"][1],
                    "choices_3": problem["choices"][2],
                    "choices_4": problem["choices"][3],
                    "choices_5": problem["choices"][4],
                })
            else:
                prompt = prompt_base
                runnable = prompt | model | StrOutputParser()
                answer = runnable.invoke({
                    "question": problem["question"],
                    "paragraph": paragraph_text,
                    "choices_1": problem["choices"][0],
                    "choices_2": problem["choices"][1],
                    "choices_3": problem["choices"][2],
                    "choices_4": problem["choices"][3],
                    "choices_5": problem["choices"][4],
                })

            answer_list.append([paragraph['id'], i, problem['answer'], answer,
                                problem['score']])  # id, problem_num, gt_answer, pred, score
            save_result_pd(save_path, answer_list)


@click.command()
@click.option('--test_file', help='test file path')
@click.option('--save_path', help='save path')
@click.option('--model_name', help='choice between gpt-4, llama-2, palm, kt')
@click.option('--start_num', default=0, help='evaluation start to this number')
@click.option('--end_num', default=50, help='evaluation end to this number')
def main(test_file, save_path, model_name, start_num, end_num):
    main_func(test_file, save_path, model_name, start_num, end_num,
              prompt_base=active_prompt, prompt_plus=active_prompt_plus)


if __name__ == "__main__":
    main()
