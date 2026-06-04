from __future__ import annotations


class CharTokenizer:
    """A tiny character tokenizer for educational language modeling."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        if not chars:
            raise ValueError("Tokenizer needs at least one character.")
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as exc:
            raise ValueError(f"Character {exc.args[0]!r} was not in the training vocabulary.") from exc

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def state_dict(self) -> dict[str, dict]:
        return {"stoi": self.stoi}

    @classmethod
    def from_state_dict(cls, state: dict[str, dict]) -> "CharTokenizer":
        obj = cls.__new__(cls)
        obj.stoi = dict(state["stoi"])
        obj.itos = {i: ch for ch, i in obj.stoi.items()}
        obj.vocab_size = len(obj.stoi)
        return obj
