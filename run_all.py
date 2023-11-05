from main import main_func
import os
import click


@click.command()
@click.option('--dir_name', help='directory name for save file')
@click.option('--save_name', help='save file suffix')
@click.option('--model_name', help='model name')
def run_all(dir_name, save_name, model_name):
    for year in range(2015, 2024):
        test_file_name = os.path.join('data', f'{year}_11_KICE.json')
        save_path = os.path.join('result', dir_name, f'{year}_11_KICE_{save_name}.csv')
        main_func(test_file_name, save_path, model_name)


if __name__ == '__main__':
    run_all()
