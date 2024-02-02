import json
import os
import openai
import requests
from dotenv import load_dotenv

class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val

    def execute(self, completion_request):
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            "X-NCP-APIGW-API-KEY": self._api_key_primary_val,
            'Content-Type': 'application/json; charset=utf-8',
        }
        return requests.post(self._host + '/testapp/v1/chat-completions/HCX-003', headers=headers, json=completion_request).text



def basic_prompt(model, paragraph, question, choices, question_plus="", no_paragraph=False):
    system_prompt = """
        국어 시험 문제를 푸는 똑똑한 학생으로써 다음 문제의 답을 구하세요.
        지문을 읽고, 질문에 대한 답을 1부터 5까지의 선택지 중에 한 개만 골라서 대답해야 합니다.
    """
    if not no_paragraph:
        user_prompt = f"""
            지문 :
            {paragraph}
        """
    else:
        user_prompt = ""
    if question_plus:
        user_prompt += f"""
            <보기> :
            {question_plus}
        """
    user_prompt += f"""
        질문 :
        {question}

        선택지 :
        1번 - {choices[0]}
        2번 - {choices[1]}
        3번 - {choices[2]}
        4번 - {choices[3]}
        5번 - {choices[4]}

        1번, 2번, 3번, 4번, 5번 중에 하나를 정답으로 고르세요. 정답 :
    """

    completion = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return completion.usage.prompt_tokens, completion.usage.completion_tokens ,completion.choices[0].message.content


def talk_prompt(model, paragraph, question, choices, question_plus="", no_paragraph=False):
    system_prompt = """
        국어 시험 문제를 푸는 대한민국의 고3 수험생으로서 위의 요약을 바탕으로 다음 문제의 답을 구하세요.

         문제를 풀이할 때, 반드시 지문을 참고하세요.
         문제는 무조건 1개의 정답만 있습니다.
         문제를 풀이할 때 모든 선택지들을 검토하세요.
         모든 선택지마다 근거를 지문에서 찾아 설명하세요.

         다음의 형식을 따라 답변하세요.

         1번: "(지문 속 근거가 된 문장)" + (자세한 풀이) + (선택지 1번에 대한 답변)
         2번: "(지문 속 근거가 된 문장)" + (자세한 풀이) + (선택지 2번에 대한 답변)
         3번: "(지문 속 근거가 된 문장)" + (자세한 풀이) + (선택지 3번에 대한 답변)
         4번: "(지문 속 근거가 된 문장)" + (자세한 풀이) + (선택지 4번에 대한 답변)
         5번: "(지문 속 근거가 된 문장)" + (자세한 풀이) + (선택지 5번에 대한 답변)
         최종 정답: (최종 정답)
    """
    if not no_paragraph:
        user_prompt = f"""
            지문 :
            {paragraph}
        """
    else:
        user_prompt = ""
    if question_plus:
        user_prompt += f"""
            이 문제는 아래와 같이 <보기>가 주어져 있습니다. 문제의 각 선택지들을 해결하기 위한 배경 지식을 설명해 주고 있는 것이 <보기>로써, 각 선택지들을 지문과 연결시키고, <보기>의 지식을 활용하면 각 선택지의 참과 거짓을 판단할 수 있습니다.
            문제를 해결할 때, 반드시 <보기>의 내용을 이용해서 문제를 해결해야 합니다.
            <보기> :
            {question_plus}
        """
    user_prompt += f"""
        질문 :
        {question}

        선택지 :
        1번 - {choices[0]}
        2번 - {choices[1]}
        3번 - {choices[2]}
        4번 - {choices[3]}
        5번 - {choices[4]}

    """

    completion = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], top_p=0)
    return completion.usage.prompt_tokens, completion.usage.completion_tokens ,completion.choices[0].message.content


def literature_prompt(model, paragraph, question, choices, question_plus="", no_paragraph=False):
    system_prompt = """
        - 대한민국의 대학수학능력시험의 국어 시험 문제를 푸는 수험생으로서 위의 요약을 바탕으로 다음 문제의 답을 구하세요.

        - 문제를 풀이할 때, 반드시 지문을 참고하세요.
        - 문제는 무조건 1개의 정답만 있습니다.
        - 문제를 풀이할 때 모든 선택지들을 검토하세요.
        - 모든 선택지마다 근거를 지문에서 찾아 설명하세요.
        - 답을 모르겠어도 꼭 답변을 작성하세요.

        !!반드시 다음의 형식대로만 답변하세요.
        1번: (선택지 1번에 대한 답변) + "(지문 속 근거가 된 문장)"
        2번: (선택지 2번에 대한 답변) + "(지문 속 근거가 된 문장)"
        3번: (선택지 3번에 대한 답변) + "(지문 속 근거가 된 문장)"
        4번: (선택지 4번에 대한 답변) + "(지문 속 근거가 된 문장)"
        5번: (선택지 5번에 대한 답변) + "(지문 속 근거가 된 문장)"
        최종 정답: (최종 정답)
    """
    if not no_paragraph:
        user_prompt = f"""
            지문 :
            {paragraph}
        """
    else:
        user_prompt = ""
    if question_plus:
        user_prompt += f"""
            이 문제는 아래와 같이 <보기>가 주어져 있습니다. 문제의 각 선택지들을 해결하기 위한 배경 지식을 설명해 주고 있는 것이 <보기>로써, 각 선택지들을 지문과 연결시키고, <보기>의 지식을 활용하면 각 선택지의 참과 거짓을 판단할 수 있습니다.
            문제를 해결할 때, 반드시 <보기>의 내용을 이용해서 문제를 해결해야 합니다.
            <보기> :
            {question_plus}
        """
    user_prompt += f"""
        질문 :
        {question}

        선택지 :
        1번 - {choices[0]}
        2번 - {choices[1]}
        3번 - {choices[2]}
        4번 - {choices[3]}
        5번 - {choices[4]}

    """
    if model == "HCX-003":
        
        # API KEY 불러오기
        load_dotenv()
        naver_api_key = os.environ["NAVER_API_KEY"]
        naver_gateway_key = os.environ["NAVER_GATEWAY_KEY"]
        if not naver_api_key or not naver_gateway_key:
            raise ValueError("NAVER API KEY empty!")
        
        # API 요청 객체 생성
        completion_executor = CompletionExecutor(
            host='https://clovastudio.apigw.ntruss.com',
            api_key=naver_api_key,
            api_key_primary_val=naver_gateway_key
        )

        # API 요청
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        completion = completion_executor.execute({
            'messages': messages,
            'topP': 0.6,
            'topK': 40,
            'maxTokens': 1024,
            'temperature': 0.2,
            'repeatPenalty': 5.0,
            'stopBefore': [],
            'includeAiFilters': False,
            'seed': 0
        })
        completion = json.loads(completion)
        result = completion.get('result', {})
        input_length = result.get('inputLength', 0)
        output_length = result.get('outputLength', 0)
        message_content = result.get('message', {}).get('content', '')
        return input_length, output_length, message_content
    else:
        completion = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], top_p=0)
        return completion.usage.prompt_tokens, completion.usage.completion_tokens ,completion.choices[0].message.content


def grammar_prompt(model, paragraph, question, choices, question_plus="", get_prompt=False, no_paragraph=False):
    system_prompt = """
        - 대한민국의 대학수학능력시험의 국어 시험 문제를 푸는 수험생으로서 위의 요약을 바탕으로 다음 문제의 답을 구하세요.

        - '지문 속 목적어의 성격'과 '선택지 속 목적어의 성격'이 서로 같은 선택지를 1개만 고르세요.
        - 모두 같은 선택지는 무조건 1개만 존재합니다.
        - 답을 모르겠어도 꼭 답변을 작성하세요.
        - 문제를 풀이할 때 5개의 모든 선택지를 검토하세요.
        - 자료나 돈처럼 실제 손으로 만질 수 있는 것은 '실제적인 단어'입니다.
        - 관심, 집중, 인기 이론처럼, 실제 손으로 만질 수 없는 것은 '추상적인 단어'입니다.

        !!반드시 다음의 형식대로만 답변하세요.
        1번: - 지문 속 동사ⓐ의 목적어: "(목적어)" + 지문 속 목적어의 성격 : "(실제적인 단어 or 추상적인 단어)"
             - 선택지 속 동사ⓐ의 목적어: "(목적어)" + 선택지 속 목적어의 성격 : "(실제적인 단어 or 추상적인 단어)"
        2번: - 지문 속 동사ⓑ의 목적어: "(목적어)" + 지문 속 목적어의 성격 : "(실제적인 단어 or 추상적인 단어)"
             - 선택지 속 동사ⓑ의 목적어: "(목적어)" + 선택지 속 목적어의 성격 : "(실제적인 단어 or 추상적인 단어)"
        3번: - 지문 속 동사ⓒ의 목적어: "(목적어)" + 지문 속 목적어의 성격 : "(실제적인 단어 or 추상적인 단어)"
             - 선택지 속 동사ⓒ의 목적어: "(목적어)" + 선택지 속 목적어의 성격 : "(실제적인 단어 or 추상적인 단어)"
        4번: - 지문 속 동사ⓓ의 목적어: "(목적어)" + 지문 속 목적어의 성격 : "(실제적인 단어 or 추상적인 단어)"
             - 선택지 속 동사ⓓ의 목적어: "(목적어)" + 선택지 속 목적어의 성격 : "(실제적인 단어 or 추상적인 단어)"
        5번: - 지문 속 동사ⓔ의 목적어: "(목적어)" + 지문 속 목적어의 성격 : "(실제적인 단어 or 추상적인 단어)"
             - 선택지 속 동사ⓔ의 목적어: "(목적어)" + 선택지 속 목적어의 성격 : "(실제적인 단어 or 추상적인 단어)"
        최종 정답: 지문 속 목적어와 선택지 속 목적어의 성격이 서로 같은 선택지는 "(최종 정답)"입니다.
    """
    if not no_paragraph:
        user_prompt = f"""
            지문 :
            {paragraph}
        """
    else:
        user_prompt = ""
    if question_plus:
        user_prompt += f"""
            이 문제는 아래와 같이 <보기>가 주어져 있습니다. 문제의 각 선택지들을 해결하기 위한 배경 지식을 설명해 주고 있는 것이 <보기>로써, 각 선택지들을 지문과 연결시키고, <보기>의 지식을 활용하면 각 선택지의 참과 거짓을 판단할 수 있습니다.
            문제를 해결할 때, 반드시 <보기>의 내용을 이용해서 문제를 해결해야 합니다.
            <보기> :
            {question_plus}
        """
    user_prompt += f"""
        질문 :
        {question}
        선택지 :
        1번 - {choices[0]}
        2번 - {choices[1]}
        3번 - {choices[2]}
        4번 - {choices[3]}
        5번 - {choices[4]}
    """
    if get_prompt:
        return system_prompt +"\n\n" +user_prompt
    
    if model == "HCX-003":
        
        # API KEY 불러오기
        load_dotenv()
        naver_api_key = os.environ["NAVER_API_KEY"]
        naver_gateway_key = os.environ["NAVER_GATEWAY_KEY"]
        if not naver_api_key or not naver_gateway_key:
            raise ValueError("NAVER API KEY empty!")
        
        # API 요청 객체 생성
        completion_executor = CompletionExecutor(
            host='https://clovastudio.apigw.ntruss.com',
            api_key=naver_api_key,
            api_key_primary_val=naver_gateway_key
        )

        # API 요청
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        completion = completion_executor.execute({
            'messages': messages,
            'topP': 0.6,
            'topK': 40,
            'maxTokens': 1024,
            'temperature': 0.2,
            'repeatPenalty': 5.0,
            'stopBefore': [],
            'includeAiFilters': False,
            'seed': 0
        })
        completion = json.loads(completion)
        result = completion.get('result', {})
        input_length = result.get('inputLength', 0)
        output_length = result.get('outputLength', 0)
        message_content = result.get('message', {}).get('content', '')
        return input_length, output_length, message_content
    else:
        completion = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], top_p=0)
        return completion.usage.prompt_tokens, completion.usage.completion_tokens ,completion.choices[0].message.content


'''
literature_prompt example

지문:
사람들이 지속적으로 책을 읽는 이유 중 하나는 즐거움이다. 독서의 즐거움에는 여러 가지가 있겠지만 그 중심에는 ‘소통의 즐거움’이 있다.독자는 독서를 통해 책과 소통하는 즐거움을 경험한다. 독서는필자와 간접적으로 대화하는 소통 행위이다. 독자는 자신이 속한사회나 시대의 영향 아래 필자가 속해 있거나 드러내고자 하는 사회나 시대를 경험한다. 직접 경험하지 못했던 다양한 삶을 필자를 매개로 만나고 이해하면서 독자는 더 넓은 시야로 세계를바라볼 수 있다. 이때 같은 책을 읽은 독자라도 독자의 배경지식이나 관점 등의 독자 요인, 읽기 환경이나 과제 등의 상황 요인이 다르므로, 필자가 보여 주는 세계를 그대로 수용하지 않고 저마다 소통 과정에서 다른 의미를 구성할 수 있다.[A] (이러한 소통은 독자가 책의 내용에 대해 질문하고 답을 찾아내는 과정에서 가능해진다. 독자는 책에서 답을 찾는 질문, 독자 자신에게서 답을 찾는 질문 등을 제기할 수 있다. 전자의 경우 책에 명시된 내용에서 답을 발견할 수 있고, 책의 내용들을 관계 지으며 답에 해당하는 내용을 스스로 구성할 수도 있다. 또한 후자의 경우 책에는 없는 독자의 경험에서 답을 찾을 수 있다. 이런 질문들을 풍부히 생성하고 주체적으로 답을 찾을 때 소통의 즐거움은 더 커진다.)한편 독자는 ㉠ (다른 독자와 소통하는 즐거움을 경험할 수도 있다.) 책과의 소통을 통해 개인적으로 형성한 의미를 독서 모임이나 독서 동아리 등에서 다른 독자들과 나누는 일이 이에 해당한다. 비슷한 해석에 서로 공감하며 기존 인식을 강화하거나 관점의 차이를 확인하고 기존 인식을 조정하는 과정에서, 독자는자신의 인식을 심화 확장할 수 있다. 최근 소통 공간이 온라인으로 확대되면서 독서를 통해 다른 독자들과 소통하며 즐거움을누리는 양상이 더 다양해지고 있다. 자신의 독서 경험을 담은 글이나 동영상을 생산 공유함으로써, 책을 읽지 않은 타인이 책과 소통하도록 돕는 것도 책을 통한 소통의 즐거움을 나누는 일이다.

질문:
윗글의 내용과 일치하지 않는 것은?

선택지:
1번 - 같은 책을 읽은 독자라도 서로 다른 의미를 구성할 수 있다.
2번 - 다른 독자와의 소통은 독자가 인식의 폭을 확장하도록 돕는다.
3번 - 독자는 직접 경험해 보지 못했던 다양한 삶을 책의 필자를 매개로 접할 수 있다.
4번 - 독자의 배경지식, 관점, 읽기 환경, 과제는 독자의 의미 구성에 영향을 주는 독자 요인이다.
5번 - 독자는 책을 읽을 때 자신이 속한 사회나 시대의 영향을 받으며 필자와 간접적으로 대화한다.

1번: "같은 책을 읽은 독자라도 독자의 배경지식이나 관점 등의 독자 요인, 읽기 환경이나 과제 등의 상황 요인이 다르므로, 필자가 보여 주는 세계를 그대로 수용하지 않고 저마다 소통 과정에서 다른 의미를 구성할 수 있다."라는 문장에서 확인할 수 있습니다.
2번: "비슷한 해석에 서로 공감하며 기존 인식을 강화하거나 관점의 차이를 확인하고 기존 인식을 조정하는 과정에서, 독자는자신의 인식을 심화 확장할 수 있다."라는 문장에서 확인할 수 있습니다.
3번: "직접 경험하지 못했던 다양한 삶을 필자를 매개로 만나고 이해하면서 독자는 더 넓은 시야로 세계를바라볼 수 있다."라는 문장에서 확인할 수 있습니다.
4번: 지문에서는 "독자의 배경지식이나 관점 등의 독자 요인, 읽기 환경이나 과제 등의 상황 요인"이라고 언급하고 있지만, 이들이 "독자의 의미 구성에 영향을 주는 독자 요인"이라고 명시적으로 언급하고 있지 않습니다.
5번: "독자는 자신이 속한사회나 시대의 영향 아래 필자가 속해 있거나 드러내고자 하는 사회나 시대를 경험한다."라는 문장에서 확인할 수 있습니다.
최종 정답: 4번
'''


'''
grammar_prompt example

지문:
법령의 조문은 대개 'A에 해당하면 B를 해야 한다.'처럼 요건과효과로 구성된 조건문으로 규정된다. 하지만 그 요건이나 효과가항상 일의적인 것은 아니다. 법조문에는 구체적 상황을 고려해야그 상황에 ⓐ(맞는) 진정한 의미가 파악되는 불확정 개념이 사용될 수 있기 때문이다. 개인 간 법률관계를 규율하는 민법에서 불확정 개념이 사용된 예로 ‘손해 배상 예정액이 부당히 과다한경우에는 법원은 적당히 감액할 수 있다.’라는 조문을 ⓑ(들) 수 있다. 이때 법원은 요건과 효과를 재량으로 판단할 수 있다. 손해배상 예정액은 위약금의 일종이며, 계약 위반에 대한 제재인 위약벌도 위약금에 속한다. 위약금의 성격이 둘 중 무엇인지 증명되지 못하면 손해 배상 예정액으로 다루어진다.채무자의 잘못으로 계약 내용이 실현되지 못하여 계약 위반이발생하면, 이로 인해 손해를 입은 채권자가 손해 액수를 증명해야 그 액수만큼 손해 배상금을 받을 수 있다. 그러나 손해 배상 예정액이 정해져 있었다면 채권자는 손해 액수를 증명하지 않아도 손해 배상 예정액만큼 손해 배상금을 받을 수 있다. 이때 손해 액수가 얼마로 증명되든 손해 배상 예정액보다 더 받을 수는 없다. 한편 위약금이 위약벌임이 증명되면 채권자는 위약벌에 해당하는 위약금을 ⓒ(받을) 수 있고, 손해 배상 예정액과는 달리 법원이 감액할 수 없다. 이때 채권자가 손해 액수를증명하면 손해 배상금도 받을 수 있다.불확정 개념은 행정 법령에도 사용된다. 행정 법령은 행정청이구체적 사실에 대해 행하는 법 집행인 행정 작용을 규율한다. 법령상 요건이 충족되면 그 효과로서 행정청이 반드시 해야 하는특정 내용의 행정 작용은 기속 행위이다. 반면 법령상 요건이 충족되더라도 그 효과인 행정 작용의 구체적 내용을 ⓓ(고를)수 있는 재량이 행정청에 주어져 있을 때, 이러한 재량을 행사하는 행정 작용은 재량 행위이다. 법령에서 불확정 개념이 사용되면 이에 근거한 행정 작용은 대개 재량 행위이다.행정청은 재량으로 재량 행사의 기준을 명확히 정할 수 있는데 이 기준을 ㉠(재량 준칙)이라 한다. 재량 준칙은 법령이 아니므로 재량 준칙대로 재량을 행사하지 않아도 근거 법령 위반은 아니다. 다만 특정 요건하에 재량 준칙대로 특정한 내용의 적법한 행정 작용이 반복되어 행정 관행이 생긴 후에는, 같은 요건이 충족되면 행정청은 동일한 내용의 행정 작용을 해야 한다. 행정청은 평등 원칙을 ⓔ(지켜야) 하기 때문이다.

질문:
문맥상 ⓐ～ⓔ의 의미와 가장 가까운 것은?

선택지:
1번 - 이것이 네가 찾는 자료가 ⓐ(맞는지) 확인해 보아라.
2번 - 그 부부는 노후 대책으로 적금을 ⓑ(들고) 안심했다.
3번 - 그의 파격적인 주장은 학계의 큰 주목을 ⓒ(받았다).
4번 - 형은 땀 흘려 울퉁불퉁한 땅을 평평하게 ⓓ(골랐다).
5번 - 그분은 우리에게 한 약속을 반드시 ⓔ(지킬) 것이다.

1번: ⓐ는 '맞는'을 의미하며, 선택지 1번은 '확인해 보아라'라는 의미로 ⓐ와 의미가 일치하지 않습니다. "(그 상황에 ⓐ(맞는) 진정한 의미가 파악되는 불확정 개념이 사용될 수 있기 때문이다.)"
2번: ⓑ는 '들 수 있다'를 의미하며, 선택지 2번은 '안심했다'라는 의미로 ⓑ와 의미가 일치하지 않습니다. "('손해 배상 예정액이 부당히 과다한경우에는 법원은 적당히 감액할 수 있다.'라는 조문을 ⓑ(들) 수 있다.)"
3번: ⓒ는 '받을 수 있다'를 의미하며, 선택지 3번은 '주목을 받았다'라는 의미로 ⓒ와 의미가 일치하지 않습니다. "(채권자는 위약벌에 해당하는 위약금을 ⓒ(받을) 수 있고, 손해 배상 예정액과는 달리 법원이 감액할 수 없다.)"
4번: ⓓ는 '고를 수 있다'를 의미하며, 선택지 4번은 '평평하게 골랐다'라는 의미로 ⓓ와 의미가 일치하지 않습니다. "(법령상 요건이 충족되더라도 그 효과인 행정 작용의 구체적 내용을 ⓓ(고를)수 있는 재량이 행정청에 주어져 있을 때, 이러한 재량을 행사하는 행정 작용은 재량 행위이다.)"
5번: ⓔ는 '지켜야 한다'를 의미하며, 선택지 5번은 '반드시 지킬 것이다'라는 의미로 ⓔ와 의미가 일치합니다. "(행정청은 평등 원칙을 ⓔ(지켜야) 하기 때문이다.)"
최종 정답: 5번
'''