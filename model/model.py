"""CNN encoder + Bahdanau attention + LSTM decoder, wrapped as Seq2Seq.

Consolidates the spec's encoder/attention/decoder/seq2seq into one file (the four
classes are small and read best top-to-bottom for an interview). Attention is
pad-masked: it only ever attends to real-ink feature cells, never to the white
padding that keeps a batch rectangular.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet18_Weights

from .config import EOS_ID, PAD_ID, SOS_ID


class Encoder(nn.Module):
    """ResNet-18 backbone (avgpool+fc removed) -> ((B, H'*W', enc), (H', W'))."""

    def __init__(self, encoder_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = torchvision.models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.projection = nn.Conv2d(512, encoder_dim, kernel_size=1)

    def forward(self, images):
        feats = self.backbone(images)                 # (B,512,H',W')
        feats = self.projection(feats)                # (B,enc,H',W')
        B, C, H, W = feats.shape
        seq = feats.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B,HW,enc)
        return seq, (H, W)


class BahdanauAttention(nn.Module):
    """Additive attention: score = v^T tanh(W_e e + W_d h), with optional pad mask."""

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden, mask=None):
        att1 = self.encoder_att(encoder_out)                        # (B,HW,att)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)        # (B,1,att)
        scores = self.full_att(torch.tanh(att1 + att2)).squeeze(2)  # (B,HW)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)                # padding -> ~0 weight
        alpha = F.softmax(scores, dim=1)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)     # (B,enc)
        return context, alpha


class Decoder(nn.Module):
    """LSTMCell decoder with attention; one LaTeX token per timestep."""

    def __init__(self, vocab_size, embedding_dim, encoder_dim, decoder_dim,
                 attention_dim, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.lstm = nn.LSTMCell(embedding_dim + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.fc = nn.Linear(decoder_dim + encoder_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, encoder_out, mask=None):
        """Mean over VALID cells only (padding excluded) -> initial (h, c)."""
        if mask is None:
            mean = encoder_out.mean(dim=1)
        else:
            m = mask.unsqueeze(2).float()
            mean = (encoder_out * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.init_h(mean), self.init_c(mean)

    def step(self, token, h, c, encoder_out, mask=None):
        """One decode step. token:(B,) -> logits:(B,V), alpha:(B,HW), (h,c)."""
        emb = self.embedding(token)                             # (B,emb)
        context, alpha = self.attention(encoder_out, h, mask)   # (B,enc),(B,HW)
        h, c = self.lstm(torch.cat([emb, context], dim=1), (h, c))
        logits = self.fc(self.dropout(torch.cat([h, context], dim=1)))
        return logits, alpha, (h, c)

    def forward(self, encoder_out, captions, mask=None, teacher_forcing_ratio=1.0):
        """captions:(B,T) with <SOS>..<EOS>. Returns logits:(B,T-1,V), alphas:(B,T-1,HW)."""
        B, T = captions.shape
        h, c = self.init_hidden(encoder_out, mask)
        logits_all, alphas = [], []
        token = captions[:, 0]
        for t in range(1, T):
            logits, alpha, (h, c) = self.step(token, h, c, encoder_out, mask)
            logits_all.append(logits)
            alphas.append(alpha)
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            token = captions[:, t] if use_teacher else logits.argmax(1)
        return torch.stack(logits_all, 1), torch.stack(alphas, 1)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, cfg, pretrained=True):
        super().__init__()
        self.encoder = Encoder(cfg["encoder_dim"], pretrained)
        self.decoder = Decoder(vocab_size, cfg["embedding_dim"], cfg["encoder_dim"],
                               cfg["decoder_dim"], cfg["attention_dim"], cfg["dropout"])

    @staticmethod
    def build_mask(valid_hw, in_h, in_w, feat_h, feat_w, device):
        """Map per-image valid (h,w) pixels -> boolean mask over the feature grid.
        Ratio-based (no fragile conv arithmetic). Returns (B, feat_h*feat_w) or None."""
        if valid_hw is None:
            return None
        B = valid_hw.size(0)
        vh = (valid_hw[:, 0].float() / in_h * feat_h).ceil().clamp(1, feat_h).long()
        vw = (valid_hw[:, 1].float() / in_w * feat_w).ceil().clamp(1, feat_w).long()
        rows = torch.arange(feat_h, device=device).view(1, feat_h, 1)
        cols = torch.arange(feat_w, device=device).view(1, 1, feat_w)
        m = (rows < vh.view(B, 1, 1)) & (cols < vw.view(B, 1, 1))
        return m.view(B, feat_h * feat_w)

    def forward(self, images, captions, valid_hw=None, teacher_forcing_ratio=1.0):
        feats, (fh, fw) = self.encoder(images)
        mask = self.build_mask(valid_hw, images.size(2), images.size(3), fh, fw, images.device)
        return self.decoder(feats, captions, mask, teacher_forcing_ratio)

    @torch.no_grad()
    def greedy(self, images, valid_hw=None, max_len=200):
        """Greedy decode -> list of id-lists. For quick sanity / cheap val accuracy."""
        self.eval()
        feats, (fh, fw) = self.encoder(images)
        mask = self.build_mask(valid_hw, images.size(2), images.size(3), fh, fw, images.device)
        B = feats.size(0)
        h, c = self.decoder.init_hidden(feats, mask)
        token = torch.full((B,), SOS_ID, dtype=torch.long, device=feats.device)
        done = torch.zeros(B, dtype=torch.bool, device=feats.device)
        seqs = [[] for _ in range(B)]
        for _ in range(max_len):
            logits, _, (h, c) = self.decoder.step(token, h, c, feats, mask)
            token = logits.argmax(1)
            for i in range(B):
                if done[i]:
                    continue
                if token[i].item() == EOS_ID:
                    done[i] = True
                else:
                    seqs[i].append(token[i].item())
            if done.all():
                break
        return seqs
