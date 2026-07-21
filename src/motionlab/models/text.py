"""Frozen T5 text encoding with word->frame timing for the transcript slot,
and pooled caption encoding for the action slot."""
import torch

_tok, _enc = None, None


def _load(model_name: str, device: str):
    global _tok, _enc
    if _enc is None:
        from transformers import AutoTokenizer, T5EncoderModel
        _tok = AutoTokenizer.from_pretrained(model_name)
        _enc = T5EncoderModel.from_pretrained(model_name).to(device).eval()
        for p in _enc.parameters():
            p.requires_grad_(False)
    return _tok, _enc


@torch.no_grad()
def encode_transcripts(word_lists: list, window: int, model_name: str = "google/flan-t5-base",
                       device: str = "cuda", max_tokens: int = 64):
    """word_lists: per sample [(start_frame, end_frame, word), ...].
    Tokenizes word-by-word so each token gets its word's center-frame time in [0,1].
    Returns emb (B,N,768), times (B,N), pad_mask (B,N) True=pad. Empty lists OK."""
    tok, enc = _load(model_name, device)
    B = len(word_lists)

    all_ids, all_times = [], []
    for words in word_lists:
        ids, times = [], []
        for s, e, w in words:
            wids = tok(w, add_special_tokens=False).input_ids
            center = ((s + e) / 2) / max(1, window)
            ids.extend(wids)
            times.extend([center] * len(wids))
        all_ids.append(ids[:max_tokens])
        all_times.append(times[:max_tokens])

    N = max(1, max(len(i) for i in all_ids))
    input_ids = torch.zeros(B, N, dtype=torch.long, device=device)
    attn = torch.zeros(B, N, dtype=torch.long, device=device)
    times = torch.zeros(B, N, device=device)
    for b, (ids, ts) in enumerate(zip(all_ids, all_times)):
        if ids:
            input_ids[b, :len(ids)] = torch.tensor(ids, device=device)
            attn[b, :len(ids)] = 1
            times[b, :len(ts)] = torch.tensor(ts, device=device)

    emb = enc(input_ids=input_ids, attention_mask=attn).last_hidden_state
    pad_mask = attn == 0
    # fully-empty rows would make cross-attention NaN; leave one unmasked zero token
    empty = pad_mask.all(dim=1)
    if empty.any():
        pad_mask[empty, 0] = False
        emb[empty] = 0
    return emb, times, pad_mask


@torch.no_grad()
def encode_captions(captions: list[str], model_name: str = "google/flan-t5-base",
                    device: str = "cuda") -> torch.Tensor:
    """Mean-pooled caption embeddings (B,768). Empty strings -> zeros."""
    tok, enc = _load(model_name, device)
    batch = tok([c or " " for c in captions], return_tensors="pt", padding=True,
                truncation=True, max_length=48).to(device)
    h = enc(**batch).last_hidden_state
    mask = batch.attention_mask[..., None].float()
    pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
    for i, c in enumerate(captions):
        if not c:
            pooled[i] = 0
    return pooled
