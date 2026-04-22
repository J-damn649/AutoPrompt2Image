import torch
import torch.nn.functional as F
import clip
import json
class StyleExtractor:
    def __init__(self, device="cuda"):
        self.device = device
        self.model, _ = clip.load("ViT-B/32", device=device)
        self.style_vocab =json.load(open("/root/autodl-tmp/llama-diffusion/dataset/style_vocab.json", "r"))
        self.vocab_embeds = self._build_vocab_embeddings()

    def _build_vocab_embeddings(self):
        tokens = clip.tokenize(self.style_vocab).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features  # [V, D]

    #  RAG → style embedding
    def build_style_embedding(self, rag_results):
        embeds = []
        weights = []

        for r in rag_results:
            emb = torch.tensor(r["embedding"]).to(self.device)
            emb = emb / emb.norm()

            embeds.append(emb)
            weights.append(r["score"])

        embeds = torch.stack(embeds)  # [K, D]
        weights = torch.tensor(weights).to(self.device)

        weights = weights / weights.sum()

        style_emb = (embeds * weights.unsqueeze(-1)).sum(dim=0)
        style_emb = style_emb / style_emb.norm()

        return style_emb  # [D]

    #  embedding → vocab
    def embedding_to_style(self, style_emb, topk=5):
        style_emb = style_emb.float()
        vocab_embeds = self.vocab_embeds.float()
        sims = torch.matmul(style_emb,vocab_embeds.T)  # [V]

        topk_idx = sims.topk(topk).indices.tolist()

        styles = [self.style_vocab[i] for i in topk_idx]

        return ", ".join(styles)
    
    def extract(self, rag_results, topk=5):
        style_emb = self.build_style_embedding(rag_results)
        style_text = self.embedding_to_style(style_emb, topk=topk)
        return style_text