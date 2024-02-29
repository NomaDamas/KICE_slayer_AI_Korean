import click
import os

from main import main_func
from prompts import test_prompt, test_prompt_plus


@click.command()
@click.option('--name', help='save name to this test')
def main(name):
    run_list_2015 = [1, 7, 8, 11, 35, 36]
    run_list_2016 = [11, 40]
    run_list_2017 = [6, 11, 12, 13, 29, 34]
    prompt_base = test_prompt
    prompt_plus = test_prompt_plus
    for year, run_list in zip(range(2015, 2018), [run_list_2015, run_list_2016, run_list_2017]):
        test_file_name = os.path.join('data', f'{year}_11_KICE.json')
        save_path = os.path.join('result', 'train_prompting', f'{year}_11_KICE_{name}.csv')
        main_func(test_file_name, save_path, 'gpt-4', run_list=run_list, prompt_base=prompt_base,
                  prompt_plus=prompt_plus)


if __name__ == '__main__':
    main()
