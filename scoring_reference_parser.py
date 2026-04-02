import requests
import os
from dotenv import load_dotenv

load_dotenv()

file_dir = "/Users/jinminseong/Desktop/수능 확정 등급컷/ocr_image"
file_lst = ['2022_table.png', '2023_table.png', '2024_table.png']


url = "https://api.upstage.ai/v1/document-ai/document-parse"
headers = {"Authorization": f"Bearer {os.getenv('UPSTAGE_API_KEY')}"}

def image_paser():
    html_result = []
    for file in file_lst:
        files = {"document": open(os.path.join(file_dir, file), "rb")}
        response = requests.post(url, headers=headers, files=files)
        html_result.append(response.json()['content']['html'])

    return html_result

if __name__ == '__main__':
    test = image_paser()
    tt = 0
