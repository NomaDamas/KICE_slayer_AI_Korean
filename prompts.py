import openai


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
        국어 시험 문제를 푸는 대한민국의 고3 수험생으로서 위의 요약을 바탕으로 다음 문제의 답을 구하세요.

         문제를 풀이할 때, 반드시 지문을 참고하세요.
         문제는 무조건 1개의 정답만 있습니다.
         문제를 풀이할 때 모든 선택지들을 검토하세요.
         모든 선택지마다 근거를 지문에서 찾아 설명하세요.

         다음의 형식을 따라 답변하세요.
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

    completion = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], top_p=0)
    return completion.usage.prompt_tokens, completion.usage.completion_tokens ,completion.choices[0].message.content


def grammar_prompt(model, paragraph, question, choices, question_plus="", get_prompt=False, no_paragraph=False):
    system_prompt = """
        당신은 국어 시험 문제를 푸는 대한민국의 고3 수험생으로서 최종 정답을 고르시오.

        '지문 속 목적어의 성격'과 '선택지 속 목적어의 성격'이 서로 같은 선택지를 1개만 고르세요.
        모두 같은 선택지는 무조건 1개만 존재합니다.

        문제를 풀이할 때 5개의 모든 선택지를 검토하세요.

        자료나 돈처럼 실제 손으로 만질 수 있는 것은 '실제적인 단어'입니다.
        관심, 집중, 인기 이론처럼, 실제 손으로 만질 수 없는 것은 '추상적인 단어'입니다.

        다음의 형식대로만 답변하세요.
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
        최종 정답: (지문 속 목적어와 선택지 속 목적어의 성격이 서로 같은 선택지는 "(최종 정답)"입니다.
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
    completion = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], top_p=0)
    return completion.usage.prompt_tokens, completion.usage.completion_tokens ,completion.choices[0].message.content
