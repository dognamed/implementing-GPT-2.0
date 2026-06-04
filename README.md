# Implementing GPT-2.0

Karpathy의 `nn-zero-to-hero`, `makemore`, GPT 강의 노트 스타일을 바탕으로 만든 교육용 GPT-2 구현입니다.  
복잡한 프레임워크 없이 PyTorch만 사용해서 transformer language model의 핵심 구조를 직접 볼 수 있게 구성했습니다.

## 핵심 아이디어

GPT-2는 이전 토큰들만 보고 다음 토큰을 예측하는 autoregressive language model입니다.

이 저장소의 구현은 다음 요소를 포함합니다.

- token embedding과 positional embedding
- masked multi-head self-attention
- residual connection과 layer normalization
- GELU MLP block
- causal language modeling loss
- weight tying: token embedding과 출력 projection 가중치 공유
- GPT-2 논문/구현에서 쓰이는 residual projection 초기화 방식

## 프로젝트 구조

```text
.
├── data/
│   └── input.txt          # 작은 학습 예제 텍스트
├── gpt2/
│   ├── __init__.py
│   ├── model.py           # GPT-2 모델 본체
│   └── tokenizer.py       # 문자 단위 토크나이저
├── scripts/
│   ├── sample.py          # 저장된 모델에서 텍스트 생성
│   └── train.py           # 학습 스크립트
├── tests/
│   └── test_model.py      # 핵심 shape/causal/generation 테스트
└── requirements.txt
```

## 설치

PyTorch 설치 안정성을 위해 Python 3.10-3.12 환경을 권장합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 학습 실행

CPU에서도 빠르게 확인할 수 있도록 기본 설정은 아주 작은 모델입니다.

```bash
python scripts/train.py --max-iters 300
```

학습이 끝나면 `out/ckpt.pt`가 저장됩니다.

## 텍스트 생성

```bash
python scripts/sample.py --checkpoint out/ckpt.pt --prompt "GPT"
```

## 구현 설명

자세한 구조 설명은 [docs/EXPLANATION.md](docs/EXPLANATION.md)에 정리했습니다.

## 더 큰 모델로 실험하기

작은 예제 대신 직접 텍스트를 넣고 모델 크기를 키울 수 있습니다.

```bash
python scripts/train.py \
  --data data/input.txt \
  --n-layer 6 \
  --n-head 6 \
  --n-embd 384 \
  --block-size 128 \
  --batch-size 32 \
  --max-iters 5000
```

## 참고자료 반영

- `nn-zero-to-hero`: gradient-based learning을 작은 단위부터 직접 구현하며 이해하는 접근을 따랐습니다.
- `makemore`: character-level language modeling, batching, sampling 흐름을 단순하고 투명하게 가져왔습니다.
- GPT/GPT-2 Colab 계열 강의: transformer block, causal self-attention, residual stream, logits/loss/generation 루프를 직접 구현하는 방식으로 반영했습니다.

이 구현은 OpenAI의 전체 GPT-2 체크포인트를 재현하는 목적이 아니라, GPT-2의 구조와 학습 원리를 과제 수준에서 명확하게 설명하고 실험할 수 있도록 만든 compact implementation입니다.
