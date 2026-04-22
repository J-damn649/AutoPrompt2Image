import torch
import faiss
import os
import numpy as np
import open_clip

class CLIPRAG:
    def __init__(self, style_dir="../dataset/style_db"):

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )

        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        self.style_texts = []
        self.style_names = []

        for file in os.listdir(style_dir):
            name = file.replace(".txt", "")
            with open(os.path.join(style_dir, file), "r") as f:
                text = f.read().strip()
                self.style_texts.append(text)
                self.style_names.append(name)

        with torch.no_grad():
            tokens = self.tokenizer(self.style_texts)
            embeddings = self.model.encode_text(tokens).cpu().numpy()
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        self.embeddings = embeddings

    def retrieve(self, user_input, top_k=3):
        tokens = self.tokenizer([user_input])

        with torch.no_grad():
            query = self.model.encode_text(tokens).cpu().numpy()
            query = query / np.linalg.norm(query, axis=1, keepdims=True)

        D, I = self.index.search(query, top_k)

        sim = 1 / (D + 1e-6)
        weights = sim / sim.sum()

        results = []

        for i, w in zip(I[0], weights[0]):
            style_text = self.style_db[i]
            results.append((style_text, float(w)))

        return results