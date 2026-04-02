import ast
import os
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import logging
from pathlib import Path
from typing import List
import re
from openai import AsyncOpenAI
from autorag.utils.util import get_event_loop, process_batch
from pydantic import BaseModel


def kice_metric(
        metric_inputs,
        model: str = "gpt-4o-mini-2024-07-18",
        batch_size: int = 5,
) -> List[int]:
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    loop = get_event_loop()
    tasks = [
        async_kice_metric(client, metric_input['generation_gt'], metric_input['generated_texts'], model)
        for idx, metric_input in metric_inputs.iterrows()
    ]
    results = loop.run_until_complete(process_batch(tasks, batch_size=batch_size))
    return results


async def async_kice_metric(
        client,
        generation_gt: List[str],
        pred: str,
        model: str = "gpt-4o-mini-2024-07-18",
) -> int:
    class Response(BaseModel):
        choice: int
    # TODO: autorag없이 한버전에서 필요함
    generation_gt = ast.literal_eval(generation_gt['generation_gt'][0])

    # parse the generation_gt
    choice_gt = int(generation_gt[0].split("(")[0])
    right_score = int(generation_gt[0].split("(")[1].split(")")[0])
    assert choice_gt in [1, 2, 3, 4, 5], "The choice_gt must be in [1, 2, 3, 4, 5]."
    assert right_score in [2, 3], "The right score must be 2 or 3."

    # get the response from the model
    completion = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system",
             "content": "당신은 시험을 채점하는 채점관입니다. 학생의 대답을 보고, 학생이 몇 번을 선택하였는지 구분하세요. 모든 답변은 1~5번 중에 하나입니다. 학생이 선택한 답변을 반환하세요. 만약 학생이 답변을 하지 못했다면 0번을 반환하세요."},
            {"role": "user",
             "content": "해당 문제는 동건이의 콧수염에 대하여 묻는 문제입니다. 동건이는 콧수염이 있지만, 그 길이가 예전에 비해 길지 않으므로 가장 적합한 선택지는 3번입니다."},
            {"role": "assistant", "content": "3"},
            {"role": "user", "content": pred},
        ],
        response_format=Response,
    )

    user_choice = completion.choices[0].message.parsed.choice

    if user_choice == choice_gt:
        return right_score
    else:
        return 0


def split_exam(df):
    df['qid_for_merge'] = df['qid'].apply(lambda x: x[:4])
    merged_df = df.groupby('qid_for_merge').agg({
        'kice_metric': list
    }).reset_index()
    return merged_df


def separate_scoring(df):
    def separate(annual_row_lst):
        common_problem_score = 0
        choice_problem_score = 0
        test_cnt = 0

        for idx, score in enumerate(annual_row_lst):
            if idx < 34:
                common_problem_score += score
                test_cnt += 1
            else:
                choice_problem_score += score
        assert test_cnt == 34

        return {'common_problem_score': common_problem_score, 'choice_problem_score': choice_problem_score,
                'overall_sum': common_problem_score + choice_problem_score}

    df['score_result'] = df.apply(
        lambda row: {'common_problem_score': sum(row['kice_metric']), 'choice_problem_score': None,
                     'overall_sum': sum(row['kice_metric'])} if int(
            row['qid_for_merge']) < 2022 else separate(row['kice_metric']), axis=1)
    result = pd.concat([df, df['score_result'].apply(pd.Series)], axis=1)
    result = result.rename(columns={0: 'common_problem_score', 1: 'choice_problem_score'})
    return result


def main(exp_num: str):
    generated_answer_lst = []
    best_file = None

    project_file_dir = f"autorag_project_dir/{exp_num}/prompt_node_line/generator"
    all_files = os.listdir(project_file_dir)

    for file in all_files:
        if 'best' in file and file.endswith('.parquet'):
            best_file = file
            break
    assert best_file != None

    experience_summary = pd.read_csv(os.path.join(project_file_dir, 'summary.csv'))
    # best_file 로드로 gt와 qid 가져오기
    completion_df = pd.read_parquet(os.path.join(project_file_dir, best_file), engine='fastparquet')

    # TODO: Apply async for calculate
    for idx, file_name in enumerate(all_files):
        if re.match(r'^\d+\.parquet$', file_name):
            filtered_model_name = \
                experience_summary.loc[experience_summary['filename'] == file_name, 'module_params'].iloc[0]
            filtered_model_name = ast.literal_eval(filtered_model_name)

            if 'model' in filtered_model_name:
                filtered_model_name = filtered_model_name['model']
            else:
                filtered_model_name = filtered_model_name['llm']

            model_name = filtered_model_name.split('/')[-1]

            # TODO: kice metric기준으로 하기 지금은 임시로 없앰
            # df_result = pd.read_parquet(os.path.join(project_file_dir, file_name), engine='fastparquet').drop(
            #     columns=['kice_metric'])
            # 그냥한 버전
            df_result = pd.read_parquet(os.path.join(project_file_dir, file_name), engine='fastparquet').drop(columns=['qid'])
            df_result = pd.concat([completion_df[['qid', 'generation_gt']], df_result], axis=1)

            # Scoring each model result.
            generated_answer = kice_metric(metric_inputs=df_result)
            df_result = pd.concat([df_result, pd.DataFrame(generated_answer, columns=['kice_metric'])], axis=1)
            logging.info(f"최종 점수:{sum(df_result['kice_metric'])}")
            df_result.to_csv(f'scored_result_{model_name}.csv', index=False)

            # Save Scoring result customized to KO-SAT
            generated_answer_lst.append(df_result)
            df_result = split_exam(df_result)
            overall = separate_scoring(df_result)
            overall.to_csv(f"scoring_result/{model_name}.csv", index=False)


if __name__ == "__main__":
    # TODO: 실험 일괄 체점하는 코드 완성시키기 -> 실험번호의 Range를 설정
    model_lst = ['108_gemini-3-pro-preview']

    for model in model_lst:
        main(model)
        logging.info(f"모델 {model} 채점완료!")
