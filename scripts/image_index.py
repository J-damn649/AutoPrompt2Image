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

        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        print("Image index built successfully:", len(self.image_paths))

    def encode_image(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feature = self.model.encode_image(image)
        
        feature = feature.cpu().numpy()
        feature = feature / np.linalg.norm(feature, axis=1, keepdims=True)
        
        return feature



    def search_embedding(self, query_embedding, top_k=3):
        """
        query_embedding: numpy array shape (1, dim)
        """

        # 🔥 确保归一化（非常关键）
        query_embedding = query_embedding / np.linalg.norm(
            query_embedding, axis=1, keepdims=True
        )

        # 🔍 FAISS 检索
        D, I = self.index.search(query_embedding.astype("float32"), top_k)

        results = []

        for idx, score in zip(I[0], D[0]):
            results.append({
                "path": self.image_paths[idx],
                "score": float(score)
            })

        return results