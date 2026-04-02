import anthropic

from timeout_solve_run import ReasoningRUN
import google.generativeai as genai
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI()
genai.configure()
antropic_client = anthropic.Anthropic()

PROMTPT17 = """
당신은 비판적인 사고력을 가진 시험 감독관 입니다. 문제와 선택지를 보고 철학과 교수의 입장, 그리고 실제 정답을 보고, 철학과 교수의 의견이 옳은지 틀린지 알려주세요. 
그리고 그 판단을 한 근거를 제시하시오. 옳은지에 대한 여부를 판단하기 위해 문제를 이해하고 계획을 세워보세요 그리고 이를 알기 위해 계획에 따라 단계별로 해결하세요. 
확신이 있을때까지 여러번의 검토를 거치십시오.


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
윗글을 바탕으로 <보기>를 이해한 반응으로 가장 적절한 것은? [3점]
    
선택지 :
1번 - 롱게네스의 견해에 의하면, 프로그램으로 재현된 의식만으로 인격이 될 수 있다는 갑의 입장은 옳겠군.
2번 - 스트로슨의 견해에 의하면, 신체를 지니지 않은 존재에게 인격이 귀속될 수 없다는 을의 입장은 옳지 않겠군.
3번 - 칸트 이전까지 유력했던 견해에 의하면, ‘생각하는 나’의 지속만으로는 인격의 동일성을 보장하지 않는다는 갑의 입장은 옳지 않겠군.
4번 - 칸트의 견해에 의하면, 인격의 통시적 동일성은 그것에 대한 가정이 선행될 필요 없이 사고 기능의 동일성을 통해 판단 된다는 을의 입장은 옳겠군.
5번 - 롱게네스의 견해에 의하면, 인간과 상이한 존재에 의해서도 동일하게 수행될 수 있는 사고 기능이 인격의 동일성을 판단하는 기준이라는 을의 입장은 옳겠군.

<보 기>
갑: 두뇌에서 일어나는 의식을 스캔하여 프로그램으로 재현
한다고 상상해 보자. 그런 경우, 본래의 자신과 재현된 
의식은 동일한 인격이 아니야. 두뇌에서 일어나는 의식은 
신체 전체의 기여로 일어난 것이기 때문이지. 즉, 프로그
램으로 재현된 의식은 인격일 수 없어. ‘생각하는 나’의 
지속만으로는 인격의 동일성이 보장될 수 없고, 살아 
있는 신체도 인격의 구성 요소에 포함되어야 하거든.
을: 그렇지 않아. 프로그램으로 재현된 의식은 본래의 자신과 
동일한 인격이야. 비록 프로그램은 신체가 없지만 우리 
두뇌와 프로그램이 수행하는 사고 기능에는 근본적인 
차이가 없거든. 인격의 동일성은 어떤 가정도 두지 않고 
이러한 사고 기능의 동일성만을 기준으로 판단해야 해.


평가원이 공개한 정답은 3번인 ‘칸트 이전까지 유력했던 견해에 의하면 ’생각하는 나‘의 지속만으로는 인격의 동일성을 보장하지 않는다는 갑의 입장은 옳지 않겠군’이다.
그렇지만 이 문제를 본 철학과 교수는 문제의 정답은 없다면서 다음과 같은 주장을 해.
그러나 이 교수는 갑의 입장은 옳기에 3번이 정답이 될 수 없다고 주장했다.
지문을 보면 ‘칸트 이전까지 인격의 동일성을 설명하는 유력한 견해는 생각하는 나인 영혼이 단일한 주관으로서 시간의 흐름 속에 지속한다는 것이었다’는 문장이 지문 도입부에 나온다.
그런데 스캔 프로그램으로 의식이 재현되면 ‘단일한 주관’이라는 조건을 충족하지 않기 때문에 ‘생각하는 나의 지속만으로는 인격의 동일성을 보장하지 않는다’는 갑의 입장은 옳다는 것이다.
또한 이 교수는 “개체 a와 b 그리고 속성 C에 대해 ‘a=b이고 a가 C면, b도 C다’를 통해 풀 수 있는 문제라 생각할 수 있지만, 얼핏 당연해 보이는 이 풀이는 실제로는 잘못된 풀이”라고 말했다.
그는 “갑은 ‘생각하는 나’에 대해서 말하고 있지 영혼에 대해서는 말하고 있지 않아서, ‘생각하는 나’와 ‘영혼’의 연결 고리가 필요하다”며 “이 둘의 유일한 연결고리는 ‘생각하는 나인 영혼’이라는 표현인데 지문과 보기 어디에도 나오지 않는 표현”이라고 했다.
평가원의 정답과 철학과 교수의 의견중 어떤 의견이 더 합당한지 말하고 그에 따른 근거를 지문과 문제, 그리고 평가원의 정답과 대조되는 철학과 교수의 의견을 기반으로 제시하시오.


합당한 부분:
근거:
"""

def save_claude_thinking(user_prompt: str, output_file: str = None):
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"claude_thinking_{timestamp}.txt"

    response = antropic_client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
        },
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )

    # 파일에 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Claude Sonnet 4.5 - Extended Thinking\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"📝 프롬프트:\n{user_prompt}\n\n")
        f.write("=" * 60 + "\n\n")

        # Thinking 블록과 텍스트 블록 분리
        for block in response.content:
            if block.type == "thinking":
                f.write("🤔 내부 추론 과정:\n")
                f.write("-" * 60 + "\n")
                f.write(f"{block.thinking}\n\n")
            elif block.type == "text":
                f.write("💬 최종 답변:\n")
                f.write("-" * 60 + "\n")
                f.write(f"{block.text}\n\n")

        # 메타데이터
        f.write("=" * 60 + "\n")
        f.write("📊 메타데이터:\n")
        f.write(f"모델: {response.model}\n")
        f.write(f"입력 토큰: {response.usage.input_tokens}\n")
        f.write(f"출력 토큰: {response.usage.output_tokens}\n")

    print(f"✅ 저장 완료: {output_file}")
    return output_file


def save_gemini_thinking(user_prompt: str, output_file: str = None):
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"gemini_thinking_{timestamp}.txt"

    model = genai.GenerativeModel('gemini-2.5-pro')

    # 추론 과정을 명시적으로 요청
    enhanced_query = (
            user_prompt
    )

    response = model.generate_content(enhanced_query)

    # 파일에 저장

    with open(output_file, 'w', encoding='utf-8') as f:
        for part in response.candidates[0].content.parts:
            if part.thought:  # thinking 블록
                f.write("🤔 내부 추론 과정 (Thought):\n")
                f.write("-" * 60 + "\n")
                f.write(f"{part.text}\n\n")
            elif part.text and not part.thought:  # 일반 텍스트
                f.write("💬 최종 답변:\n")
                f.write("-" * 60 + "\n")
                f.write(f"{part.text}\n\n")

        f.write("💭 응답 (추론 과정 포함):\n")
        f.write("-" * 60 + "\n")
        f.write(f"{response.text}\n\n")

        # 메타데이터 (사용 가능한 경우)
        if hasattr(response, 'usage_metadata'):
            f.write("=" * 60 + "\n")
            f.write("📊 메타데이터:\n")
            f.write(f"입력 토큰: {response.usage_metadata.prompt_token_count}\n")
            f.write(f"출력 토큰: {response.usage_metadata.candidates_token_count}\n")
            f.write(f"총 토큰: {response.usage_metadata.total_token_count}\n")

    print(f"✅ 저장 완료: {output_file}")
    return output_file


def save_openai_reasoning_summary(user_prompt: str, model: str = "o3-2025-04-16", output_file: str = None):
    """
    OpenAI reasoning 모델의 추론 요약본을 txt 파일로 저장
    model: "gpt-5", "gpt-5-mini", "o4-mini" 등
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"openai_reasoning_{timestamp}.txt"

    # ⭐ Responses API 사용 (새로운 API)
    response = openai_client.responses.create(
        model=model,
        reasoning={
            "effort": "high",  # low, medium, high
            "summary": "auto"  # auto, detailed, concise
        },
        input=[
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    # 파일에 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"OpenAI {model} - Reasoning Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"📝 프롬프트:\n{user_prompt}\n\n")
        f.write(f"⚙️  추론 강도: high\n\n")
        f.write("=" * 60 + "\n\n")

        # output 배열 순회
        for item in response.output:
            if item.type == "reasoning":
                f.write("🤔 추론 과정 요약 (Reasoning Summary):\n")
                f.write("-" * 60 + "\n")
                for summary_item in item.summary:
                    if summary_item.type == "summary_text":
                        f.write(f"{summary_item.text}\n\n")

            elif item.type == "message":
                f.write("💬 최종 답변:\n")
                f.write("-" * 60 + "\n")
                for content in item.content:
                    if content.type == "output_text":
                        f.write(f"{content.text}\n\n")

        # 토큰 사용량
        f.write("=" * 60 + "\n")
        f.write("📊 토큰 사용량:\n")
        f.write(f"입력 토큰: {response.usage.input_tokens}\n")
        f.write(f"추론 토큰: {response.usage.output_tokens_details.reasoning_tokens}\n")
        f.write(f"출력 토큰: {response.usage.output_tokens}\n")
        f.write(f"총 토큰: {response.usage.total_tokens}\n")

    print(f"✅ 저장 완료: {output_file}")
    return output_file

def compare_all_models(user_prompt: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 각 모델 실행 및 저장
    claude_file = save_claude_thinking(user_prompt, f"claude_{timestamp}.txt")
    gemini_file = save_gemini_thinking(user_prompt, f"gemini_{timestamp}.txt")
    o3_file = save_openai_reasoning_summary(user_prompt, output_file=f"o3_{timestamp}.txt")

    # 통합 비교 파일 생성
    comparison_file = f"comparison_{timestamp}.txt"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("3개 모델 비교\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"프롬프트: {user_prompt}\n\n")
        f.write(f"생성된 파일:\n")
        f.write(f"- Claude: {claude_file}\n")
        f.write(f"- Gemini: {gemini_file}\n")
        f.write(f"- o3: {o3_file}\n")

    print(f"✅ 비교 파일 생성 완료: {comparison_file}")

if __name__ == "__main__":
    """
    gemini는 think pad 지원하지 않음. gpt는 인증절차가 복잡함. 클로드만 표본으로 분석진행
    """
    import pandas as pd
    reasoning_run = ReasoningRUN()
    gemini_answer = reasoning_run.gemini_run(PROMTPT17, model="gemini-3-pro-preview")
    print(gemini_answer)
    with open("gemini-3-preview.txt", 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n\n")
        f.write(f"📝 프롬프트:\n{PROMTPT17}\n\n")
        f.write(f"⚙️  추론 강도: high\n\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"{gemini_answer}")


    # df = pd.read_parquet(
    #     "/Users/jinminseong/Desktop/KICE_slayer_AI_Korean/autorag_project_dir/110_claude-sonnet-4-5-20250929/prompt_node_line/generator/best_0.parquet")


# 교수의 의견을 가지고 맞춘애들의 정정생각들을 보고 결과 알려주기 3번주장하더라에 대한 ai의 의견. 몇개에 정리하기
# -> 만점의 모델들 체크하기
# 인간 교수의 피드백받고 자기 답변 수정하는건지 보는건지 리포트거리 주기
#
# → 교수의 말의 ㄱㄱ