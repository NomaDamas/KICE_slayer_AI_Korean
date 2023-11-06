from main import main_func
import os
import click
from prompts import wook_prompt, wook_prompt_plus, ranking_prompt, ranking_prompt_plus


@click.command()
@click.option('--dir_name', help='directory name for save file')
@click.option('--save_name', help='save file suffix')
@click.option('--model_name', help='model name')
def run_all(dir_name, save_name, model_name):
    prompt_base = ranking_prompt
    prompt_plus = ranking_prompt_plus
    for year in range(2015, 2024):
        test_file_name = os.path.join('data', f'{year}_11_KICE.json')
        save_path = os.path.join('result', dir_name, f'{year}_11_KICE_{save_name}.csv')
        main_func(test_file_name, save_path, model_name, prompt_base=prompt_base, prompt_plus=prompt_plus)


if __name__ == '__main__':
    run_all()
