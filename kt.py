import json
import os
from typing import List, Optional, Any
from datetime import datetime
import hmac, hashlib

import requests
from pytz import timezone

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms import BaseLLM
from langchain.schema import LLMResult, Generation


class KTMRC(BaseLLM):
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> LLMResult:
        passages = []
        final_prompts = []
        for prompt in prompts:
            passage, final_prompt = self.__make_prompt_passage(prompt)
            passages.append(passage)
            final_prompts.append(final_prompt)
        responses = []
        for prompt, passages in zip(final_prompts, passages):
            response = self.__generate_answer(prompt, passages, *self.__get_signature())
            responses.append(response.text)

        return LLMResult(generations=[[Generation(text=response)] for response in responses])

    def __generate_answer(self, prompt, passage, client_key, signature, timestamp):
        url = "https://aiapi.genielabs.ai/kt/nlp/mrc"
        headers = {
            'x-client-key': client_key,
            'x-client-signature': signature,
            'x-auth-timestamp': timestamp,
            'Content-Type': 'application/json',
            'charset': 'utf-8'
        }
        data = {
            "query": prompt,
            "itemcnt": 1,
            "passages": [passage]
        }

        return requests.post(url, headers=headers, data=json.dumps(data))

    def __get_signature(self):
        # timestamp 생성
        timestamp = datetime.now(timezone("Asia/Seoul")).strftime("%Y%m%d%H%M%S%f")[:-3]

        client_id = os.getenv('KT_CLIENT_ID')
        client_secret = os.getenv('KT_CLIENT_SECRET')
        client_key = os.getenv('KT_CLIENT_KEY')

        # HMAC 기반 signature 생성
        signature = hmac.new(
            key=client_secret.encode("UTF-8"), msg=f"{client_id}:{timestamp}".encode("UTF-8"), digestmod=hashlib.sha256
        ).hexdigest()

        return client_key, signature, timestamp

    def __make_prompt_passage(self, original_prompt: str) -> tuple[str, str]:
        if '지문 :' not in original_prompt and '질문 :' not in original_prompt:
            raise ValueError('지문과 질문을 포함한 prompt를 입력해주세요.')
        paragraph = original_prompt.split('지문 :')[-1].split('질문 :')[0]
        if '<보기> :' in paragraph:
            question_plus = paragraph.split('<보기> :')[-1].split('선택지 :')[0]
        else:
            question_plus = ''
        return paragraph + question_plus, original_prompt.replace(paragraph, '').replace(question_plus, '')

    @property
    def _llm_type(self) -> str:
        return 'kt_mrc'
