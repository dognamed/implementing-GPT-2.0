# GPT-2.0 구현 설명

이 프로젝트는 Karpathy의 `nn-zero-to-hero`, `makemore`, GPT 강의 흐름처럼 작은 언어 모델을 직접 구성하면서 GPT-2의 핵심 원리를 이해하는 것을 목표로 합니다.

## 1. 전체 흐름

언어 모델은 토큰 시퀀스가 주어졌을 때 다음 토큰을 예측합니다.

예를 들어 입력이 다음과 같다면:

```text
GPT is a transformer
```

모델은 각 위치에서 바로 다음 문자를 맞히도록 학습됩니다.

```text
G -> P
P -> T
T -> " "
" " -> i
```

이 저장소에서는 이해를 쉽게 하기 위해 문자 단위 tokenizer를 사용했습니다. 실제 GPT-2는 BPE 계열 subword tokenizer를 사용하지만, transformer 구조와 학습 objective는 동일한 방식으로 확인할 수 있습니다.

## 2. Embedding

모델은 정수 token id를 바로 처리하지 않고 vector로 바꿉니다.

- `wte`: token embedding
- `wpe`: positional embedding

Transformer 자체는 순서 정보를 직접 알지 못하기 때문에 positional embedding을 더해 각 토큰의 위치를 알려줍니다.

## 3. Causal Self-Attention

Self-attention은 각 토큰이 다른 토큰을 참고해서 자신의 표현을 갱신하는 구조입니다.

GPT 계열 모델은 autoregressive 모델이므로 미래 토큰을 보면 안 됩니다. 그래서 lower-triangular causal mask를 사용합니다.

```python
mask = torch.tril(torch.ones(block_size, block_size))
```

이 mask 덕분에 위치 `t`의 token은 `0..t` 위치까지만 볼 수 있고, `t+1` 이후의 token은 볼 수 없습니다.

## 4. Multi-Head Attention

하나의 attention만 쓰는 대신 여러 head로 나누면 서로 다른 관점의 관계를 학습할 수 있습니다.

예를 들어 한 head는 가까운 문자를, 다른 head는 줄바꿈이나 문장 구조 같은 더 긴 패턴을 볼 수 있습니다.

구현에서는 `q`, `k`, `v`를 한 번의 linear layer로 만든 뒤 head 수만큼 reshape합니다.

## 5. Transformer Block

각 block은 다음 구조를 가집니다.

```text
x = x + attention(layer_norm(x))
x = x + mlp(layer_norm(x))
```

이 방식은 GPT-2에서 사용하는 pre-layernorm transformer block과 같은 형태입니다.

- residual connection: 깊은 네트워크에서도 gradient와 정보가 잘 흐르게 함
- layer normalization: activation scale을 안정화함
- MLP: 각 token 위치별 비선형 변환을 수행함

## 6. Loss

모델 출력 `logits`는 각 위치에서 vocabulary 전체에 대한 점수입니다.

학습은 cross entropy loss로 진행합니다.

```python
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
```

모델은 정답 token의 확률을 높이고 나머지 token의 확률을 낮추는 방향으로 업데이트됩니다.

## 7. Generation

텍스트 생성은 다음 과정을 반복합니다.

1. 현재까지의 token을 모델에 넣습니다.
2. 마지막 위치의 logits만 사용합니다.
3. temperature와 top-k로 분포를 조절합니다.
4. 다음 token을 sampling합니다.
5. 새 token을 뒤에 붙입니다.

이 과정을 반복하면 모델이 한 글자씩 텍스트를 이어 씁니다.

## 8. GPT-2와 같은 점, 단순화한 점

같은 점:

- masked multi-head self-attention
- transformer block
- residual stream
- layer normalization
- GELU MLP
- autoregressive next-token prediction
- token embedding과 output projection weight tying
- GPT-2식 residual projection 초기화

단순화한 점:

- 실제 GPT-2의 BPE tokenizer 대신 문자 tokenizer 사용
- 대규모 WebText 대신 작은 예제 텍스트 사용
- distributed training, mixed precision, checkpoint sharding 등 생략
- 모델 크기를 CPU에서도 확인 가능한 수준으로 축소

따라서 이 프로젝트는 GPT-2의 산업용 재현보다는 구조 이해와 직접 실행 가능한 교육용 구현에 초점을 둡니다.
