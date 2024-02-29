import os

import click

from main import main_func
from prompts import (wook_prompt, wook_prompt_plus, ranking_prompt, ranking_prompt_plus, attack_prompt,
                     attack_prompt_plus, zero_shot_cot_en_prompt,
                     zero_shot_cot_en_prompt_plus,
                     one_shot_prompt_plus, one_shot_prompt, emotional_prompt, emotional_prompt_plus,
                     basic_prompt, basic_prompt_plus,
                     zero_shot_cot_prompt, zero_shot_cot_prompt_plus, ps_prompt, ps_prompt_plus
                     )

prompt_list = [
    (basic_prompt, basic_prompt_plus),
    (wook_prompt, wook_prompt_plus),
    (ranking_prompt, ranking_prompt_plus),
    (attack_prompt, attack_prompt_plus),
    (zero_shot_cot_prompt, zero_shot_cot_prompt_plus),
    (zero_shot_cot_en_prompt, zero_shot_cot_en_prompt_plus),
    (one_shot_prompt, one_shot_prompt_plus),
    (emotional_prompt, emotional_prompt_plus),
    (ps_prompt, ps_prompt_plus)
]
prompt_name = [
    "basic", "wook", "ranking", "attack", "zero_shot_cot", "zero_shot_cot_en", "one_shot", "emotional", "ps"
]


@click.command()
@click.option('--dir_name', help='directory name for save file')
@click.option('--model_name', help='model name')
@click.option('--start_year', default=2015)
@click.option('--end_year', default=2024)
def run_all(dir_name, model_name, start_year, end_year):
    for prompt_pack, name in zip(prompt_list, prompt_name):
        prompt_base = prompt_pack[0]
        prompt_plus = prompt_pack[1]
        for year in range(start_year, end_year + 1):
            test_file_name = os.path.join('data', f'{year}_11_KICE.json')
            save_path = os.path.join('result', dir_name, f'{year}_11_KICE_{name}_{model_name}.csv')
            main_func(test_file_name, save_path, model_name, prompt_base=prompt_base, prompt_plus=prompt_plus)


if __name__ == '__main__':
    run_all()
