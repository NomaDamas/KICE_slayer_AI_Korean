import os
from typing import List

import click
from autorag.evaluation.generation import GENERATION_METRIC_FUNC_DICT
from autorag.evaluation.metric.util import autorag_metric_loop
from autorag.evaluator import Evaluator
from autorag.schema.metricinput import MetricInput
from autorag.utils.util import get_event_loop, process_batch
from autorag import generator_models
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from llama_index.llms.upstage import Upstage

load_dotenv()

@autorag_metric_loop(fields_to_check=["generation_gt", "generated_texts"])
def kice_metric(
        metric_inputs: List[MetricInput],
        model: str = "gpt-4o-mini-2024-07-18",
        batch_size: int = 16,
) -> List[int]:
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    loop = get_event_loop()
    tasks = [
        async_kice_metric(client, metric_input.generation_gt, metric_input.generated_texts, model)
        for metric_input in metric_inputs
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

    # parse the generation_gt
    choice_gt = int(generation_gt[0].split("(")[0])
    right_score = int(generation_gt[0].split("(")[1].split(")")[0])
    assert choice_gt in [1, 2, 3, 4, 5], "The choice_gt must be in [1, 2, 3, 4, 5]."
    assert right_score in [2, 3], "The right score must be 2 or 3."

    # get the response from the model
    completion = await client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "당신은 시험을 채점하는 채점관입니다. 학생의 대답을 보고, 학생이 몇 번을 선택하였는지 구분하세요. 모든 답변은 1~5번 중에 하나입니다. 학생이 선택한 답변을 반환하세요. 만약 학생이 답변을 하지 못했다면 0번을 반환하세요."},
            {"role": "user", "content": "해당 문제는 동건이의 콧수염에 대하여 묻는 문제입니다. 동건이는 콧수염이 있지만, 그 길이가 예전에 비해 길지 않으므로 가장 적합한 선택지는 3번입니다."},
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


@click.command()
@click.option('--qa_data_path', type=click.Path(exists=True, dir_okay=False), help='Path to QA data parquet file',
              default=os.path.join('data', 'autorag', 'qa.parquet'))
@click.option('--corpus_data_path', type=click.Path(exists=True, dir_okay=False), help='Path to corpus data parquet file',
              default=os.path.join('data', 'autorag', 'corpus.parquet'))
@click.option('--config', type=click.Path(exists=True, dir_okay=False), help='Path to config file',
              default=os.path.join('autorag_config.yaml'))
@click.option('--project_dir', type=click.Path(file_okay=False), help='Path to project directory',
              default=os.path.join('autorag_project_dir'))
def main(qa_data_path, corpus_data_path, config, project_dir):
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    GENERATION_METRIC_FUNC_DICT["kice_metric"] = kice_metric
    # submit upstage llm
    generator_models["upstage"] = Upstage
    # run evaluation
    evaluator = Evaluator(qa_data_path, corpus_data_path, project_dir)
    evaluator.start_trial(config)


if __name__ == "__main__":
    main()
