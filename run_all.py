from main import main_func
import os
import click
from prompts import (wook_prompt, wook_prompt_plus, ranking_prompt, ranking_prompt_plus, marker_prompt,
                     marker_prompt_plus, attack_prompt, attack_prompt_plus, zero_shot_cot_en_prompt, zero_shot_cot_en_prompt_plus,
                     one_shot_prompt_plus, one_shot_prompt, emotional_prompt, emotional_prompt_plus,
                     active_prompt_plus, active_prompt
                     )


@click.command()
@click.option('--dir_name', help='directory name for save file')
@click.option('--save_name', help='save file suffix')
@click.option('--model_name', help='model name')
@click.option('--start_year', default=2015)
@click.option('--end_year', default=2023)
def run_all(dir_name, save_name, model_name, start_year, end_year):
    prompt_base = one_shot_prompt
    prompt_plus = one_shot_prompt_plus
    for year in range(start_year, end_year+1):
        test_file_name = os.path.join('data', f'{year}_11_KICE.json')
        save_path = os.path.join('result', dir_name, f'{year}_11_KICE_{save_name}.csv')
        main_func(test_file_name, save_path, model_name, prompt_base=prompt_base, prompt_plus=prompt_plus)


if __name__ == '__main__':
    run_all()
