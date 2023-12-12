import json
import logging
import os

import click
import openai
from dotenv import load_dotenv
from tqdm import tqdm

from prompts import basic_prompt, literature_prompt, grammar_prompt

OPENAI_MODELS = [
    "gpt-4-1106-preview",
    "gpt-4",
    "gpt-3.5-turbo-1106",
    "gpt-3.5"
]


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
    if not openai.api_key:
        raise ValueError("OPENAI API KEY empty!")


def get_answer_one_problem(data, model: str, paragraph_num: int, problem_num: int, prompt_func: callable = basic_prompt):
    problem = data[paragraph_num]["problems"][problem_num]
    no_paragraph = False
    if "no_paragraph" in list(problem.keys()):
        no_paragraph = True
    if "question_plus" in list(problem.keys()):
        question_plus_text = problem["question_plus"]
    else:
        question_plus_text = ""
    return prompt_func(
        model=model,
        paragraph=data[paragraph_num]["paragraph"],
        question=problem["question"],
        choices=problem["choices"],
        question_plus=question_plus_text,
        no_paragraph=no_paragraph
    )


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


@click.command()
@click.option('--test_file', help='test file path')
@click.option('--save_path', help='save path')
@click.option('--model', help=f'select openAI model to use: {OPENAI_MODELS}')
def main(test_file, save_path, model):
    if not test_file:
        raise ValueError("test file not set!")
    if not save_path:
        raise ValueError("save path not set!")
    logging.basicConfig(filename=f"{save_path.split('.')[0]}_log.log", level=logging.INFO)
    set_openai_key()
    if model not in OPENAI_MODELS:
        raise ValueError(f"Unsupported openai model! Please select one of {OPENAI_MODELS}")
    test = load_test(test_file)

    total_cost = 0
    _id = 0
    with open(save_path, "w", encoding="UTF-8") as fw:
        for paragraph_index, paragraph in enumerate(test):
            prompt_func = get_prompt_by_type(int(paragraph["type"]))
            for problem_index, problem in tqdm(enumerate(paragraph["problems"]), total=len(paragraph["problems"])):
                _id += 1
                if "type" in list(problem.keys()):
                    prompt_func = get_prompt_by_type(int(problem["type"]))
                answer = None
                for _ in range(3):
                    try:
                        input_token, output_token, answer = get_answer_one_problem(test, model, paragraph_index, problem_index, prompt_func)
                        cost = cost_calc(model, input_token, output_token)
                        total_cost += cost
                        logging.info(cost)
                        logging.info(answer)
                        break
                    except Exception as e:
                        print(f"RETRY, Failed! id: {_id} exception: {str(e)}")
                if not answer:
                    print(f"RETRY FAILED id: {_id}")
                    continue
                fw.write(f"""{_id}번 문제: {problem['question']}
정답: {problem['answer']}
배점: {problem['score']}
GPT 풀이: {answer}
----------------------\n""")
                fw.flush()


if __name__ == "__main__":
    main()
