import tiktoken
from anthropic import Anthropic


class TokenCounter:
    """다양한 LLM 모델의 토큰을 세는 범용 카운터"""

    def __init__(self, model_name="gpt-4"):
        """
        Args:
            model_name: 'gpt-4', 'gpt-3.5-turbo', 'claude-3-opus', 등
        """
        self.model_name = model_name

        # OpenAI 계열 모델
        if 'gpt' in model_name.lower() or 'o1' in model_name.lower():
            self.encoding = tiktoken.encoding_for_model(model_name)
            self.counter_type = 'tiktoken'
        # Claude 계열 모델
        elif 'claude' in model_name.lower():
            self.client = Anthropic()
            self.counter_type = 'anthropic'
        else:
            # 기본값: cl100k_base 인코딩 사용
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.counter_type = 'tiktoken'

    def count_tokens(self, text):
        """
        텍스트의 토큰 수를 반환

        Args:
            text: 카운트할 텍스트

        Returns:
            int: 토큰 개수
        """
        if self.counter_type == 'tiktoken':
            tokens = self.encoding.encode(text)
            return len(tokens)
        elif self.counter_type == 'anthropic':
            return self.client.count_tokens(text)

    def count_tokens_from_messages(self, messages):
        """
        대화 메시지 리스트의 토큰 수를 반환

        Args:
            messages: [{"role": "user", "content": "..."}] 형태의 리스트

        Returns:
            int: 총 토큰 개수
        """
        if self.counter_type == 'anthropic':
            # Claude는 메시지 포맷 그대로 계산
            total = 0
            for msg in messages:
                total += self.client.count_tokens(msg.get('content', ''))
            return total
        else:
            # OpenAI 모델용 메시지 토큰 계산
            tokens_per_message = 3  # 메시지 메타데이터 오버헤드
            tokens_per_name = 1

            num_tokens = 0
            for message in messages:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(self.encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name

            num_tokens += 3  # 응답 프롬프트 오버헤드
            return num_tokens

    def count_tokens_from_file(self, file_path):
        """
        파일의 토큰 수를 반환

        Args:
            file_path: 텍스트 파일 경로

        Returns:
            int: 토큰 개수
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.count_tokens(text)

    def estimate_cost(self, input_tokens, output_tokens=0):
        """
        토큰 수 기반 비용 추정 (2025년 11월 기준)

        Args:
            input_tokens: 입력 토큰 수
            output_tokens: 출력 토큰 수

        Returns:
            float: 예상 비용 (USD) 또는 None (가격 정보 없음)
        """
        # 가격표 (per 1M tokens)
        pricing = {
            'gpt-4o': {'input': 2.50, 'output': 10.00},
            'gpt-4o-mini': {'input': 0.150, 'output': 0.600},
            'gpt-4': {'input': 30.00, 'output': 60.00},
            'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
            'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
            'claude-3-opus': {'input': 15.00, 'output': 75.00},
            'claude-3.5-sonnet': {'input': 3.00, 'output': 15.00},
            'claude-3-sonnet': {'input': 3.00, 'output': 15.00},
            'claude-3-haiku': {'input': 0.25, 'output': 1.25},
            'claude-3.7-sonnet': {'input': 3.00, 'output': 15.00},
        }

        model_key = None
        # 더 정확한 매칭을 위해 긴 키부터 확인
        for key in sorted(pricing.keys(), key=len, reverse=True):
            if key in self.model_name.lower():
                model_key = key
                break

        if not model_key:
            return None

        input_cost = (input_tokens / 1_000_000) * pricing[model_key]['input']
        output_cost = (output_tokens / 1_000_000) * pricing[model_key]['output']

        return input_cost + output_cost

