"""Beam search decoding + confidence score for a single image."""
import math

import torch

from .config import EOS_ID, SOS_ID
from .model import Seq2Seq


@torch.no_grad()
def beam_search(model, image, vocab, valid_hw=None, beam_width=5, max_len=200,
                length_penalty=0.7):
    """image:(1,3,H,W) -> (latex_string, confidence). Aspect-preserving pad is masked
    out of attention; decode length is capped so a runaway beam can't OOM."""
    model.eval()
    device = image.device
    feats, (fh, fw) = model.encoder(image)
    mask = Seq2Seq.build_mask(valid_hw, image.size(2), image.size(3), fh, fw, device)

    enc = feats.expand(beam_width, -1, -1)
    mask_b = mask.expand(beam_width, -1) if mask is not None else None
    state = model.decoder.init_hidden(enc, mask_b)
    tokens = torch.full((beam_width, 1), SOS_ID, dtype=torch.long, device=device)
    scores = torch.full((beam_width,), -1e9, device=device)
    scores[0] = 0.0                                        # single root at step 0
    finished = []                                         # (norm_score, seq, logprob, len)

    for _ in range(max_len):
        logits, _, state = model.decoder.step(tokens[:, -1], state, enc, mask_b)
        logp = torch.log_softmax(logits, dim=-1)          # (beam,V)
        cand = (scores.unsqueeze(1) + logp).view(-1)      # (beam*V,)
        top_scores, top_idx = cand.topk(beam_width)
        beam_idx = torch.div(top_idx, logp.size(1), rounding_mode="floor")
        tok_idx = top_idx % logp.size(1)
        tokens = torch.cat([tokens[beam_idx], tok_idx.unsqueeze(1)], dim=1)
        state = tuple(s[beam_idx] for s in state)
        scores = top_scores

        for b in range(beam_width):
            if tok_idx[b].item() == EOS_ID:
                seq = tokens[b, 1:-1].tolist()            # drop <SOS> and <EOS>
                length = max(1, len(seq))
                finished.append((scores[b].item() / length ** length_penalty,
                                 seq, scores[b].item(), length))
                scores[b] = -1e9                          # retire from active pool
        if bool((scores <= -1e9).all()):
            break

    if not finished:                                      # nothing hit EOS in max_len
        b = int(scores.argmax())
        seq = tokens[b, 1:].tolist()
        length = max(1, len(seq))
        finished.append((scores[b].item() / length ** length_penalty,
                         seq, scores[b].item(), length))

    _, seq, logprob, length = max(finished, key=lambda x: x[0])
    confidence = math.exp(logprob / length)               # normalized probability
    return vocab.decode(seq), confidence
