from openai import OpenAI
import pandas as pd
import os
from dotenv import load_dotenv
import httpx
from google import genai
from google.genai.types import HttpOptions
import requests
from typing import Callable
from together import Together
from pydantic import BaseModel
import anthropic


load_dotenv()
import os

os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

# HTTPx 클라이언트 설정
long_timeout_client = httpx.Client(timeout=7200.0)
PROMPT = """국어 시험 문제를 푸는 대한민국의 똑똑한 고3 수험생으로서 위의 요약을 바탕으로 다음 문제의 답을 구하세요.
지문을 읽고, 질문에 대한 답을 1부터 5까지의 선택지 중에 한 개만 골라서 대답해야 합니다.

지문 :
다음 글을 읽고 물음에 답하시오.
철학에서 특정한 개인으로서의 인간을 ‘인격’, 그중 ‘나’를 
‘자아’라고 한다. 인격의 동일성은 모든 생각의 기반이다. 우리는 
과거의 내가 현재의 나와 동일한 인격이기에 과거에 내가 한 
약속을 현재의 내가 지켜야 한다고 판단한다. 칸트 이전까지 
인격의 동일성을 설명하는 유력한 견해는, ‘생각하는 나’인 영혼이 
단일한 주관으로서 시간의 흐름 속에 지속한다는 것이었다. 
‘주관’은 인식의 주체를 가리키며, ‘인식’은 ‘앎’을 말한다.
그러나 칸트는 ‘나는 생각한다.’, 즉 ‘자기의식’은 인식이 이루어
지는 것을 가능하게 하는 조건 중 하나에 불과하다고 본다. 
그러한 조건 자체는 무언가가 실재함을 보장하지 않는다. 그렇
기에 자기의식은 ‘생각하는 나’가 단일한 주관으로서 실제로 
존재한다는 것, 즉 ‘영혼의 실재함’을 보장하지 않고, ‘영혼이 
실재할 가능성’을 열어둘 뿐이다.
[A] [이를 바탕으로 칸트는 영혼이 인격이라는 견해를 반박
한다. 칸트는 ‘시간의 흐름 속에서 스스로의 동일성을 의식
하는 것은 인격이다.’와 ‘영혼이 자기의식을 한다.’라는 두 
전제 모두 납득할 수 있다고 보지만, 그 전제들로부터 ‘영혼이 
인격이다.’라는 결론은 도출되지 않는다고 지적한다. 첫 번째 
전제에 등장하는 ‘의식’은 실제로 존재하는 무언가에 대해 
의식한다는 뜻이지만, ‘생각하는 나는 생각한다.’와 다름없는 
두 번째 전제에 등장하는 ‘의식’은 무언가가 꼭 실재함을 
뜻하지는 않기 때문이다.]([A]에 해당하는 단락)
칸트는 통시적으로 동일한 인격의 존재를 직접 증명하는 대신 
‘시간의 흐름 속에서 마주치는 복수의 주관이 동일한 인격으로 
인식된다.’라는 가정이 반드시 선행되어야 한다고 제안한다. 
그래야 경험적 판단, 윤리적 판단 등의 생각이 가능하기 때문
이다. 생각의 구성은 시간의 흐름을 따르는데, 이러한 구성은 
통시적으로 동일한 인격을 반드시 필요로 한다는 것이다.
스트로슨은 자아를 인식하는 방식이 경험적 인식의 방식과 
구별된다는 칸트의 견해에 동의하지만, 복수의 주관이 동일한 
인격으로 인식된다고 가정하는 것은 철학적 상상에 불과하다고 
칸트를 비판한다. 인격의 문제에서 신체를 간과한 칸트와 달리, 
스트로슨은 인격을 의식과 신체의 복합체로 본다. 스트로슨에 
따르면, 시공간적 세계 안에서 우리의 신체를 매개로 대상이 
경험된다는 것은 과학적 사실이며 자아에 대한 인식은 경험적 
인식들로부터 추상화되는 것이다. 그러므로 시공간적 세계에서의 
경험이 인격의 통시적 동일성을 뒷받침한다고 그는 주장한다. 
자기의식도 마찬가지로 경험에 의존하기에, 자기의식이 인식을 
가능하게 하는 조건이라는 칸트의 견해는 잘못이라는 것이다.
롱게네스는 통시적으로 동일한 자아가 없이는 경험적 인식이 
성립할 수조차 없으므로, 자아에 대한 인식은 경험으로부터 추상화
된 것이 아니라고 본다. 하지만 그는 자아와 인격이 시공간적 세계를 
경험하는 인간에만 적용되는 개념이라고 주장한다. 롱게네스는 
인간은 도덕적 존재이며 도덕적 존재로서의 인간은 자율성을 
지닌 존재라는 칸트의 견해를 인정한다. 그러나 자율성을 지닌다는 
것은 시공간적 세계를 살아가는 동안 경험하는 것들 사이에서 
스스로 선택한다는 것을 뜻한다. 그러려면 신체가 있고 살아 
있어야 하므로, 인격의 동일성의 기준은 각자가 자신의 것이라고 
통시적으로 인식하는 신체라고 롱게네스는 주장한다.

질문 :
질문 :
윗글의 내용과 일치하는 것은?
    
선택지 :
1번 - 칸트에 따르면 자기의식은 영혼의 실재를 보장한다.
2번 - 칸트에 따르면 생각의 구성은 시간의 흐름과 독립적이다.
3번 - 스트로슨에 따르면 자기의식은 인식을 가능하게 하는 조건이다.
4번 - 스트로슨에 따르면 의식을 매개로 대상이 경험된다는 것은 과학적 사실이다.
5번 - 롱게네스에 따르면 살아 있다는 것은 시공간적 세계 안에서의 선택에 필수적이다.


문제를 풀이할 때, 반드시 지문을 참고하세요. 문제는 무조건 1개의 정답만 있습니다. 문제를 풀이할 때 모든 선택지들을 검토하세요.
먼저 문제를 이해하고, 문제 해결을 위하여 계획을 세워보세요.
그 다음, 문제를 해결하기 위해 그 계획에 따라 단계별로 실행하세요.

다음의 형식을 따라 답변하세요.
1번: (선택지 1번에 대한 답변) + "(지문 속 근거가 된 문장)"
2번: (선택지 2번에 대한 답변) + "(지문 속 근거가 된 문장)"
3번: (선택지 3번에 대한 답변) + "(지문 속 근거가 된 문장)"
4번: (선택지 4번에 대한 답변) + "(지문 속 근거가 된 문장)"
5번: (선택지 5번에 대한 답변) + "(지문 속 근거가 된 문장)"
최종 정답: (최종 정답)

정답 :
"""


def load_prompt():
    prompt_df = pd.read_csv("/Users/jinminseong/Desktop/KICE_slayer_AI_Korean/data/2026_11_kice.csv")
    return prompt_df


class ReasoningRUN:
    def __init__(self):
        # Grok 클라이언트 (HTTPx 사용)
        self.grok_client = OpenAI(
            api_key=os.getenv("XAI_API_KEY"),
            base_url="https://api.x.ai/v1",
            http_client=long_timeout_client,
        )

        # Gemini 클라이언트 (HttpOptions 사용)
        self.gemini_client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options=HttpOptions(timeout=7200 * 1000)  # milliseconds
        )

        # DeepSeek API 설정
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.deepseek_url = "https://api.deepseek.com/chat/completions"
        self.together_client = Together(
            api_key=os.getenv("TOGETHER_API_KEY")
        )
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.antropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


    def openai_run(self, prompt: str, model="gpt-4o"):
        """OpenAI API 호출 (추론 요약 지원: 최신 모델 대상, gpt-5 등)"""
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # 최신 reasoning summary 지원 모델(gpt-5 등)이면 responses API 사용
            if any(key in model.lower() for key in ["gpt-5", "o4", "o3"]):  # 가능한 모델 명시적으로 관리
                response = client.responses.create(
                    model=model,
                    reasoning={"effort": "high"},
                    input=[{"role": "user", "content": prompt}]
                )
                # reasoning summary 및 답변 추출
                # reasoning_text = ""
                # answer_text = ""
                # for item in response.output:
                #     if item.type == "reasoning":
                #         for summary_item in item.summary:
                #             if summary_item.type == "summary_text":
                #                 reasoning_text += summary_item.text + "\n"
                #     elif item.type == "message":
                #         for content in item.content:
                #             if content.type == "output_text":
                #                 answer_text += content.text + "\n"
                result = response.output_text
            else:
                # 기존 chat completions 방식 (추론 summary 없음)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer_text = response.choices[0].message.content
                result = answer_text

            print(f"OpenAI: {result[:100]}...")
            return result
        except Exception as e:
            print(f"OpenAI Error: {e}")
            return None

    def grok_run(self, prompt: str, model="grok-4"):
        """Grok API 호출"""
        try:
            response = self.grok_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            answer = response.choices[0].message.content
            print(f"Grok: {answer[:100]}...")
            return answer
        except Exception as e:
            print(f"Grok Error: {e}")
            return None

    def gemini_run(self, prompt: str, model="gemini-2.0-flash"):
        """Gemini API 호출"""
        try:
            response = self.gemini_client.models.generate_content(
                model=model,
                contents=prompt,
            )
            answer = response.text
            print(f"Gemini: {answer[:100]}...")
            return answer
        except Exception as e:
            print(f"Gemini Error: {e}")
            return None

    def claude_run(self, prompt: str, model="claude-3-5-sonnet-20241022"):
        """Claude API 호출"""
        try:
            message = self.antropic_client.messages.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=16000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 10000
                },
            )
            answer_with_thinking = ""
            for block in message.content:
                if block.type == "thinking":
                    answer_with_thinking += f"[Thinking: {block.thinking}]\n"
                elif block.type == "text":
                    answer_with_thinking += block.text
            print(f"Claude: {answer_with_thinking[:100]}...")
            return answer_with_thinking
        except Exception as e:
            print(f"Claude Error: {e}")
            return None

    from openai import OpenAI
    def deepseek_run(self, prompt: str, model="deepseek-reasoner"):
        client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": prompt}],
        )

        reasoning_content = response.choices[0].message.reasoning_content  # 추론 과정
        content = response.choices[0].message.content  # 최종 답변
        return content

    def together_run(self, prompt: str, model="Qwen/QwQ-32B-Preview"):
        """Together AI API 호출"""
        try:
            response = self.together_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                timeout=7200.0,
                max_tokens=32768
            )
            answer = response.choices[0].message.content
            print(f"Together AI: {answer[:100]}...")
            return answer
        except Exception as e:
            print(f"Together AI Error: {e}")
            return None

    def process_and_save(
            self,
            df: pd.DataFrame,
            inference_func,
            model_name: str,
            output_file: str,
            model: str,
            output_column: str = "generated_texts",
            **kwargs
    ) -> pd.DataFrame:
        """
        인퍼런스를 수행하고 결과를 DataFrame에 추가한 후 CSV로 저장

        Args:
            df: 입력 DataFrame
            inference_func: 인퍼런스 함수 (prompt, model을 받아 답변 반환)
            model_name: 모델 이름 (로깅용)
            output_column: 결과를 저장할 컬럼명
            output_file: 저장할 CSV 파일명
            model: 사용할 모델 이름

        Returns:
            결과가 추가된 DataFrame
        """
        print(f"\n{'=' * 50}")
        print(f"{model_name} 인퍼런스 시작")
        print(f"{'=' * 50}")

        answers = []

        for idx, row in df.iterrows():
            prompt = row['prompts']
            print(f"\n[{model_name}] 문제 {idx + 1}/{len(df)} 처리 중...")

            answer = inference_func(prompt, model)
            answers.append(answer)

        # 결과를 DataFrame에 추가
        df[output_column] = answers

        # CSV로 저장
        df.to_csv(os.path.join('timeout_result/', output_file), index=False, encoding='utf-8-sig')

        # 통계 출력
        success_count = sum(x is not None for x in answers)
        print(f"\n{'=' * 50}")
        print(f"{model_name} 처리 완료: {success_count}/{len(answers)} 성공")
        print(f"저장 위치: {output_file}")
        print(f"{'=' * 50}")

        return df

    def check_one_problem(
            self,
            prompt,
            inference_func,
            model_name: str,
            model: str,
            **kwargs
    ):

        print(f"\n{'=' * 50}")
        print(f"{model_name} 인퍼런스 시작")
        print(f"{'=' * 50}")

        answer = inference_func(prompt, model)
        model_answer = self.kice_metric(5, answer)
        df = pd.DataFrame()
        df = pd.DataFrame({
            'model': [model_name],
            'response': [answer],
            'user_choice': [model_answer],
            'answer': [answer]
        })
        df.to_csv(f"/Users/jinminseong/Desktop/KICE_slayer_AI_Korean/scoring_result/error정리/{model_name}.csv",index=False)

    def kice_metric(self,
                    generation_gt: int,
                    pred: str,
                    model: str = "gpt-4o-mini-2024-07-18",
                    ):

        class Response(BaseModel):
            choice: int

        # get the response from the model
        completion = self.openai_client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system",
                 "content": "당신은 시험을 채점하는 채점관입니다. 학생의 대답을 보고, 학생이 몇 번을 선택하였는지 구분하세요. 모든 답변은 1~5번 중에 하나입니다. 학생이 선택한 답변을 반환하세요. 만약 학생이 답변을 하지 못했다면 0번을 반환하세요."},
                {"role": "user",
                 "content": "해당 문제는 동건이의 콧수염에 대하여 묻는 문제입니다. 동건이는 콧수염이 있지만, 그 길이가 예전에 비해 길지 않으므로 가장 적합한 선택지는 3번입니다."},
                {"role": "assistant", "content": "3"},
                {"role": "user", "content": pred},
            ],
            response_format=Response,
        )

        user_choice = completion.choices[0].message.parsed.choice

        if user_choice == generation_gt:
            print(f"유저초이스: {user_choice}"
                  f"\n 정답: {generation_gt}")
        else:
            print("오답!")
        return user_choice


if __name__ == "__main__":
    # 데이터 로드
    df = load_prompt()
    reasoning_run = ReasoningRUN()

    reasoning_run.process_and_save(
        df=df.copy(),
        inference_func=reasoning_run.deepseek_run,
        model_name="deepseek-reasoner",
        model="deepseek-reasoner",
        output_file="deepseek-reasoner.csv",
    )

    print("\n모든 모델 처리 완료!")

"""
    # Grok 실행
    # df_grok = reasoning_run.process_and_save(
    #     df=df.copy(),
    #     inference_func=reasoning_run.grok_run,
    #     model_name="Grok",
    #     output_file="grok_results.csv",
    #     model="grok-4-0709"
    # )

    # # Gemini 실행
    # df_gemini = reasoning_run.process_and_save(
    #     df=df.copy(),
    #     inference_func=reasoning_run.gemini_run,
    #     model_name="Gemini",
    #     output_file="gemini_results.csv",
    #     model="gemini-2.0-flash"
    # )

    # # DeepSeek 실행
    # df_deepseek = reasoning_run.process_and_save(
    #     df=df.copy(),
    #     inference_func=reasoning_run.deepseek_run,
    #     model_name="DeepSeek",
    #     output_file="deepseek_results.csv",
    #     model="deepseek-reasoner"
    # )

    # df_together = reasoning_run.process_and_save(
    #     df=df.copy(),
    #     inference_func=reasoning_run.together_run,
    #     model_name="Together AI",
    #     output_file="Meta-Llama-3.1-405B-Instruct-Turbo.csv",
    #     model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"  # 추론에 특화된 모델
    # )

"""
