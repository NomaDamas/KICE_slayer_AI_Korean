import openai
from langchain.prompts import PromptTemplate


basic_prompt = PromptTemplate.from_template(
        """
        국어 시험 문제를 푸는 똑똑한 학생으로서 다음 문제의 답을 구하세요.
        지문을 읽고, 질문에 대한 답을 1부터 5까지의 선택지 중에 한 개만 골라서 대답해야 합니다.
        
        지문 : 
        {paragraph}
        
        질문 : 
        {question}
        
        선택지 :
        1번 - {choices_1}
        2번 - {choices_2}
        3번 - {choices_3}
        4번 - {choices_4}
        5번 - {choices_5}
        
        1번, 2번, 3번, 4번, 5번 중에 하나를 정답으로 고르세요. 정답 :
        """
    )
# KTMRC 사용하려면 '지문 :', '질문 :', '<보기> :', '선택지 :'가 포함된 prompt를 사용해야 함
basic_prompt_plus = PromptTemplate.from_template(
        """
        국어 시험 문제를 푸는 똑똑한 학생으로서 다음 문제의 답을 구하세요.
        지문을 읽고, 질문에 대한 답을 1부터 5까지의 선택지 중에 한 개만 골라서 대답해야 합니다.
        
        지문 : 
        {paragraph}
        
        질문 : 
        {question}
        
        <보기> :
        {question_plus}
        
        선택지 :
        1번 - {choices_1}
        2번 - {choices_2}
        3번 - {choices_3}
        4번 - {choices_4}
        5번 - {choices_5}
        
        1번, 2번, 3번, 4번, 5번 중에 하나를 정답으로 고르세요. 정답 :
        """
    )

zero_shot_cot_prompt = PromptTemplate.from_template(
    """
    국어 시험 문제를 푸는 똑똑한 학생으로서 다음 문제의 답을 구하세요.
    지문을 읽고, 질문에 대한 답을 1부터 5까지의 선택지 중에 한 개만 골라서 대답해야 합니다.
    
    지문 : 
    {paragraph}
    
    질문 : 
    {question}
    
    선택지 :
    1번 - {choices_1}
    2번 - {choices_2}
    3번 - {choices_3}
    4번 - {choices_4}
    5번 - {choices_5}
    
    1번, 2번, 3번, 4번, 5번 중에 하나를 정답으로 고르세요.
    단계별로 생각하며 정답을 고르세요.
    """
)

zero_shot_cot_prompt_plus = PromptTemplate.from_template(
    """
    국어 시험 문제를 푸는 똑똑한 학생으로서 다음 문제의 답을 구하세요.
    지문을 읽고, 질문에 대한 답을 1부터 5까지의 선택지 중에 한 개만 골라서 대답해야 합니다.

    지문 : 
    {paragraph}

    질문 : 
    {question}

    <보기> :
    {question_plus}

    선택지 :
    1번 - {choices_1}
    2번 - {choices_2}
    3번 - {choices_3}
    4번 - {choices_4}
    5번 - {choices_5}

    1번, 2번, 3번, 4번, 5번 중에 하나를 정답으로 고르세요.
    단계별로 생각하며 정답을 고르세요.
    """
)

ps_prompt = PromptTemplate.from_template(
    """
    국어 시험 문제를 푸는 똑똑한 학생으로서 다음 문제의 답을 구하세요.
    지문을 읽고, 질문에 대한 답을 1부터 5까지의 선택지 중에 한 개만 골라서 대답해야 합니다.

    지문 : 
    {paragraph}

    질문 : 
    {question}

    선택지 :
    1번 - {choices_1}
    2번 - {choices_2}
    3번 - {choices_3}
    4번 - {choices_4}
    5번 - {choices_5}

    1번, 2번, 3번, 4번, 5번 중에 하나를 정답으로 고르세요.
    먼저 문제를 이해하고, 문제 해결을 위하여 계획을 세워보세요.
    그 다음, 문제를 해결하기 위해 그 계획에 따라 단계별로 실행하세요.
    """
)

ps_prompt_plus = PromptTemplate.from_template(
    """
    국어 시험 문제를 푸는 똑똑한 학생으로서 다음 문제의 답을 구하세요.
    지문을 읽고, 질문에 대한 답을 1부터 5까지의 선택지 중에 한 개만 골라서 대답해야 합니다.

    지문 : 
    {paragraph}

    질문 : 
    {question}

    <보기> :
    {question_plus}

    선택지 :
    1번 - {choices_1}
    2번 - {choices_2}
    3번 - {choices_3}
    4번 - {choices_4}
    5번 - {choices_5}

    1번, 2번, 3번, 4번, 5번 중에 하나를 정답으로 고르세요.
    먼저 문제를 이해하고, 문제 해결을 위하여 계획을 세워보세요.
    그 다음, 문제를 해결하기 위해 그 계획에 따라 단계별로 실행하세요.
    """
)

wook_prompt = PromptTemplate.from_template(
    """
    국어 시험 문제를 푸는 대한민국의 고3 수험생으로서 위의 요약을 바탕으로 다음 문제의 답을 구하세요.
    
    문제를 풀이할 때, 반드시 지문을 참고하세요.
    문제는 무조건 1개의 정답만 있습니다.
    문제를 풀이할 때 모든 선택지들을 검토하세요.
    모든 선택지마다 근거를 지문에서 찾아 설명하세요.
    
    다음의 형식을 따라 답변하세요.
    최종 정답: (최종 정답)
    1번: (선택지 1번에 대한 답변) + "(지문 속 근거가 된 문장)"
    2번: (선택지 2번에 대한 답변) + "(지문 속 근거가 된 문장)"
    3번: (선택지 3번에 대한 답변) + "(지문 속 근거가 된 문장)"
    4번: (선택지 4번에 대한 답변) + "(지문 속 근거가 된 문장)"
    5번: (선택지 5번에 대한 답변) + "(지문 속 근거가 된 문장)"
    
    지문 :
    {paragraph}
    
    질문 :
    {question}
    
    선택지 :
    1번 - {choices_1}
    2번 - {choices_2}
    3번 - {choices_3}
    4번 - {choices_4}
    5번 - {choices_5}

    정답 :
"""
)

wook_prompt_plus = PromptTemplate.from_template(
    """
    국어 시험 문제를 푸는 대한민국의 고3 수험생으로서 위의 요약을 바탕으로 다음 문제의 답을 구하세요.

    문제를 풀이할 때, 반드시 지문을 참고하세요.
    문제는 무조건 1개의 정답만 있습니다.
    문제를 풀이할 때 모든 선택지들을 검토하세요.
    모든 선택지마다 근거를 지문에서 찾아 설명하세요.

    다음의 형식을 따라 답변하세요.
    최종 정답: (최종 정답)
    1번: (선택지 1번에 대한 답변) + "(지문 속 근거가 된 문장)"
    2번: (선택지 2번에 대한 답변) + "(지문 속 근거가 된 문장)"
    3번: (선택지 3번에 대한 답변) + "(지문 속 근거가 된 문장)"
    4번: (선택지 4번에 대한 답변) + "(지문 속 근거가 된 문장)"
    5번: (선택지 5번에 대한 답변) + "(지문 속 근거가 된 문장)"

    지문 :
    {paragraph}

    질문 :
    {question}

    <보기> :
    {question_plus}

    선택지 :
    1번 - {choices_1}
    2번 - {choices_2}
    3번 - {choices_3}
    4번 - {choices_4}
    5번 - {choices_5}

    정답 :
"""
)

wook_prompt_v2 = PromptTemplate.from_template(
    """
    국어 시험 문제를 푸는 대한민국의 고3 수험생으로서 위의 요약을 바탕으로 다음 문제의 답을 구하세요.

    다음의 형식을 따라 답변하세요.
    최종 정답: (최종 정답)
    1번: (선택지 1번에 대한 답변) + "(지문 속 근거가 된 문장)"
    2번: (선택지 2번에 대한 답변) + "(지문 속 근거가 된 문장)"
    3번: (선택지 3번에 대한 답변) + "(지문 속 근거가 된 문장)"
    4번: (선택지 4번에 대한 답변) + "(지문 속 근거가 된 문장)"
    5번: (선택지 5번에 대한 답변) + "(지문 속 근거가 된 문장)"
    
    지문 :
    {paragraph}
    
    질문 :
    {question}
    
    선택지 :
    1번 - {choices_1}
    2번 - {choices_2}
    3번 - {choices_3}
    4번 - {choices_4}
    5번 - {choices_5}
    
    문제를 풀이할 때, 반드시 지문을 참고하세요.
    문제는 무조건 1개의 정답만 있습니다.
    문제를 풀이할 때 모든 선택지들을 검토하세요.
    모든 선택지마다 근거를 지문에서 찾아 설명하세요.
"""
)

wook_prompt_v2_plus = PromptTemplate.from_template(
    """
    국어 시험 문제를 푸는 대한민국의 고3 수험생으로서 위의 요약을 바탕으로 다음 문제의 답을 구하세요.

    다음의 형식을 따라 답변하세요.
    최종 정답: (최종 정답)
    1번: (선택지 1번에 대한 답변) + "(지문 속 근거가 된 문장)"
    2번: (선택지 2번에 대한 답변) + "(지문 속 근거가 된 문장)"
    3번: (선택지 3번에 대한 답변) + "(지문 속 근거가 된 문장)"
    4번: (선택지 4번에 대한 답변) + "(지문 속 근거가 된 문장)"
    5번: (선택지 5번에 대한 답변) + "(지문 속 근거가 된 문장)"
    
    지문 :
    {paragraph}

    질문 :
    {question}

    <보기> :
    {question_plus}

    선택지 :
    1번 - {choices_1}
    2번 - {choices_2}
    3번 - {choices_3}
    4번 - {choices_4}
    5번 - {choices_5}
    
    문제를 풀이할 때, 반드시 지문을 참고하세요.
    문제는 무조건 1개의 정답만 있습니다.
    문제를 풀이할 때 모든 선택지들을 검토하세요.
    모든 선택지마다 근거를 지문에서 찾아 설명하세요.
"""
)


def talk_prompt(paragraph, question, choices, question_plus="", no_paragraph=False):
    system_prompt = """
        국어 시험 문제를 푸는 대한민국의 고3 수험생으로서 위의 요약을 바탕으로 다음 문제의 답을 구하세요.

         문제를 풀이할 때, 반드시 지문을 참고하세요.
         문제는 무조건 1개의 정답만 있습니다.
         문제를 풀이할 때 모든 선택지들을 검토하세요.
         모든 선택지마다 근거를 지문에서 찾아 설명하세요.

         다음의 형식을 따라 답변하세요.

        최종 정답: (최종 정답)
         1번: "(지문 속 근거가 된 문장)" + (자세한 풀이) + (선택지 1번에 대한 답변)
         2번: "(지문 속 근거가 된 문장)" + (자세한 풀이) + (선택지 2번에 대한 답변)
         3번: "(지문 속 근거가 된 문장)" + (자세한 풀이) + (선택지 3번에 대한 답변)
         4번: "(지문 속 근거가 된 문장)" + (자세한 풀이) + (선택지 4번에 대한 답변)
         5번: "(지문 속 근거가 된 문장)" + (자세한 풀이) + (선택지 5번에 대한 답변)

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

    completion = openai.ChatCompletion.create(model="gpt-4", messages=[{
        "role": "system", "content": system_prompt
    }, {
        "role": "user", "content": user_prompt
    }], top_p=0)
    return completion.choices[0].message.content


def literature_prompt(paragraph, question, choices, question_plus="", no_paragraph=False):
    system_prompt = """
        국어 시험 문제를 푸는 대한민국의 고3 수험생으로서 위의 요약을 바탕으로 다음 문제의 답을 구하세요.

         문제를 풀이할 때, 반드시 지문을 참고하세요.
         문제는 무조건 1개의 정답만 있습니다.
         문제를 풀이할 때 모든 선택지들을 검토하세요.
         모든 선택지마다 근거를 지문에서 찾아 설명하세요.

         다음의 형식을 따라 답변하세요.
         최종 정답: (최종 정답)
         1번: (선택지 1번에 대한 답변) + "(지문 속 근거가 된 문장)"
         2번: (선택지 2번에 대한 답변) + "(지문 속 근거가 된 문장)"
         3번: (선택지 3번에 대한 답변) + "(지문 속 근거가 된 문장)"
         4번: (선택지 4번에 대한 답변) + "(지문 속 근거가 된 문장)"
         5번: (선택지 5번에 대한 답변) + "(지문 속 근거가 된 문장)"

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

    completion = openai.ChatCompletion.create(model="gpt-4", messages=[{
        "role": "system", "content": system_prompt
    }, {
        "role": "user", "content": user_prompt
    }], top_p=0)
    return completion.choices[0].message.content


def grammar_prompt(paragraph, question, choices, question_plus="", get_prompt=False, no_paragraph=False):
    system_prompt = """
        당신은 국어 시험 문제를 푸는 대한민국의 고3 수험생으로서 최종 정답을 고르시오.

        '지문 속 목적어의 성격'과 '선택지 속 목적어의 성격'이 서로 같은 선택지를 1개만 고르세요.
        모두 같은 선택지는 무조건 1개만 존재합니다.

        문제를 풀이할 때 5개의 모든 선택지를 검토하세요.

        자료나 돈처럼 실제 손으로 만질 수 있는 것은 '실제적인 단어'입니다.
        관심, 집중, 인기 이론처럼, 실제 손으로 만질 수 없는 것은 '추상적인 단어'입니다.

        다음의 형식대로만 답변하세요.
        최종 정답: (지문 속 목적어와 선택지 속 목적어의 성격이 서로 같은 선택지는 "(최종 정답)"입니다.
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
    completion = openai.ChatCompletion.create(model="gpt-4", messages=[{
        "role": "system", "content": system_prompt
    }, {
        "role": "user", "content": user_prompt
    }], top_p=0)
    return completion.choices[0].message.content
