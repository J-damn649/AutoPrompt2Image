import torch
import clip
from PIL import Image
import numpy as np
import faiss
import os
class ClipImageRAG:
    def __init__(self, base_dir="/root/autodl-tmp/llama-diffusion/dataset/style_images"):

        self.image_paths = []
        self.embeddings = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.tokenizer = clip.tokenize
        for style in os.listdir(base_dir):
            style_dir = os.path.join(base_dir, style)
            if not os.path.isdir(style_dir):
                continue
            
            for img_name in os.listdir(style_dir):
                path = os.path.join(style_dir, img_name)
                try:
                    emb = self.encode_image(path)
                    self.embeddings.append(emb[0])
                    self.image_paths.append(path)
                except Exception as e:
                    print(f"Error processing {path}: {e}")

        self.embeddings = np.array(self.embeddings).astype("float32")

        #  cosine similarity
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        print("Image index construction completed", len(self.image_paths))

    def encode_image(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feature_embedding = self.model.encode_image(image)
        
        feature_embedding = feature_embedding.cpu().numpy()
        feature_embedding = feature_embedding / np.linalg.norm(feature_embedding, axis=1, keepdims=True)
        
        return feature_embedding
    def encode_text(self, text):
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            feature_embedding = self.model.encode_text(tokens)
        feature_embedding = feature_embedding.cpu().numpy()
        feature_embedding = feature_embedding / np.linalg.norm(
            feature_embedding, axis=1, keepdims=True
        )
        return feature_embedding
    def search_style(self, query_text, top_k=3):
        """
        query_embedding: numpy array shape (1, dim)
        """
        query_embedding = self.encode_text(query_text)

        query_embedding = query_embedding / np.linalg.norm(
            query_embedding, axis=1, keepdims=True
        )

        D, I = self.index.search(query_embedding.astype("float32"), top_k)

        rag_results = []

        for idx, score in zip(I[0], D[0]):
            rag_results.append({
                "path": self.image_paths[idx],
                "score": float(score),
                'embedding': self.embeddings[idx]
            })

        return rag_results