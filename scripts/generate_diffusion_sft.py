import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import torch
import random
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
from tqdm import tqdm
import time
from typing import List

DEEPSEEK_API_KEY = "#"  
IMAGE_DIR = "/root/autodl-tmp/llama-diffusion/dataset/vangogh_data"                        
OUTPUT_FILE = "/root/autodl-tmp/llama-diffusion/dataset/image_prompts.jsonl"              
START_INDEX = 0                                 
END_INDEX = 399                                
BATCH_SIZE = 10                                  
MAX_RETRIES = 3                                 
#Vangogh style keywords list (200+ styles, can be expanded as needed)
STYLE_VOCAB = [
    "oil painting",
    "thick brushstrokes",
    "warm tone",
    "vibrant colors",
    "neon lights",
    "realistic photography",
    "wet texture",
    "cinematic",
    "ink wash painting",
    "minimalist",
    "elegant lines",
    "monochrome",
    "watercolor",
    "soft pastel",
    "dreamy glow",
    "gentle texture",
    "sci-fi",
    "3D render",
    "cosmic glow",
    "ultra realistic",
    "japanese anime",
    "soft shading",
    "hand drawn",
    "dark fantasy",
    "baroque",
    "dramatic shadow",
    "stone texture",
    "nordic style",
    "natural light",
    "clean",
    "steampunk",
    "brass metal",
    "rust texture",
    "industrial",
    "impressionism",
    "brush strokes",
    "bright tone",
    "warm glow",
    "gufeng",
    "ethereal glow",
    "silk texture",
    "traditional art",
    "8-bit pixel",
    "retro game",
    "nostalgic",
    "pixelated",
    "close-up photography",
    "sharp focus",
    "vivid",
    "wasteland",
    "dystopian",
    "gritty texture",
    "dark tone",
    "cartoon",
    "kawaii",
    "soft color",
    "simple background",
    "cute",
    "landscape photography",
    "cold tone",
    "wide angle",
    "vintage oil painting",
    "elegant",
    "classical",
    "glossy metal",
    "LED glow",
    "3D",
    "chinese ink art",
    "serene",
    "pop art",
    "high saturation",
    "retro vibe",
    "clean lines",
    "realistic armor",
    "dramatic light",
    "epic",
    "surrealism",
    "blue tone",
    "glowing scales",
    "soft light",
    "street photography",
    "film grain",
    "candid",
    "fantasy",
    "vintage leather",
    "candle light",
    "mystical runes",
    "ukiyo-e",
    "traditional",
    "flat color",
    "classic japanese art",
    "realistic landscape",
    "cyber gufeng",
    "neon katana",
    "futuristic",
    "wildlife photography",
    "cold light",
    "arctic",
    "classic european",
    "warm light",
    "wooden texture",
    "cozy",
    "anime cel shading",
    "glitter",
    "bright color",
    "dramatic landscape",
    "fiery color",
    "epic composition",
    "cinematic light",
    "minimalist portrait",
    "soft focus",
    "simple",
    "brass texture",
    "retro",
    "meticulous painting",
    "delicate lines",
    "sci-fi fantasy",
    "glowing plants",
    "purple sky",
    "1950s vintage",
    "neon glow",
    "high fantasy",
    "green tone",
    "detailed costume",
    "macro photography",
    "moody tone",
    "imperial gufeng",
    "red gold",
    "grand architecture",
    "low poly",
    "geometric art",
    "pastel color",
    "minimalist 3D",
    "modern",
    "noir photography",
    "dim light",
    "black white",
    "grainy",
    "atmospheric",
    "mythical art",
    "fiery glow",
    "dynamic pose",
    "romantic landscape",
    "dark cyberpunk",
    "digital glow",
    "rustic",
    "high fashion",
    "sharp",
    "cartoon fantasy",
    "majestic",
    "gothic horror",
    "dark elegant",
    "mysterious",
    "nature macro",
    "fresh green",
    "80s retro",
    "pixel vibe",
    "chinese landscape",
    "ethereal mist",
    "grand",
    "cosmic fantasy",
    "starry texture",
    "vintage warm",
    "cozy photography",
    "rustic texture",
    "feudal japan",
    "dark stealth",
    "cinematic",
    "fantasy landscape",
    "sunlight glow",
    "black white photography",
    "high contrast",
    "artistic",
    "fantasy geology",
    "glowing gems",
    "reflective",
    "magical",
    "ins photography",
    "bright tone",
    "vintage sci-fi",
    "gear details",
    "floral painting",
    "rich color",
    "detailed texture",
    "military sci-fi",
    "gritty",
    "post-impressionism",
    "chinese classical",
    "3D cartoon",
    "smooth texture",
    "adorable",
    "alpine landscape",
    "clear light",
    "vintage cabaret",
    "glamorous",
    "fairy tale",
    "whimsical",
    "tiny details",
    "city photography",
    "night neon",
    "reflection",
    "purple glow",
    "ancient ruins",
    "food photography",
    "white background",
    "norse medieval",
    "rugged wood",
    "stormy sea",
    "historical",
    "soft aesthetic",
    "gradient color",
    "peaceful",
    "smooth",
    "sci-fi anime",
    "white tone",
    "archaeological photography",
    "weathered",
    "textured",
    "day of the dead",
    "intricate patterns",
    "festive",
    "modern minimalist",
    "clean composition",
    "japanese yokai",
    "glowing tails",
    "mystical",
    "sci-fi utopia",
    "glass metal",
    "glowing light",
    "golden light",
    "vintage carnival",
    "fantasy elemental",
    "mossy texture",
    "rugged",
    "minimalist home",
    "lacquer texture",
    "intricate details",
    "fantasy nature",
    "night scene",
    "modern corporate office",
    "professional",
    "floral landscape",
    "sunlight",
    "vintage still life",
    "dark stealth",
    "silhouette",
    "mirror landscape",
    "clear light",
    "k-pop anime",
    "spotlight",
    "high quality",
    "historical fantasy",
    "forge fire",
    "warm glow",
    "cyberpunk street",
    "wet reflection",
    "traditional",
    "golden scales",
    "dynamic",
    "intricate",
    "line art",
    "plain background",
    "hard sci-fi",
    "metal texture",
    "japanese matsuri",
    "festive",
    "winter fantasy",
    "ice texture",
    "blue glow",
    "cozy still life",
    "nautical fantasy",
    "weathered wood",
    "storm waves",
    "macro delicate",
    "comic book art",
    "bold color",
    "high contrast",
    "asian landscape",
    "misty",
    "gothic romance",
    "pale glow",
    "classic still life",
    "gundam anime",
    "metallic armor",
    "moody landscape",
    "green tone",
    "retro photography",
    "film grain",
    "greek mythology",
    "wet texture",
    "classical",
    "middle eastern landscape",
    "silhouette",
    "street art",
    "bold lines",
    "dark background",
    "winter cartoon",
    "round shape",
    "classical theater",
    "rich color",
    "rustic fantasy",
    "mossy texture",
    "urban landscape",
    "golden glow",
    "transparent texture",
    "historical architecture",
    "ancient texture",
    "synthwave",
    "vaporwave",
    "neon grid",
    "retro 80s",
    "wildlife photo",
    "gentle",
    "natural green",
    "bohemian",
    "creative",
    "norse mythology",
    "lightning",
    "storm",
    "muscular",
    "japanese food photography",
    "green tone",
    "asian festival",
    "golden glow",
    "sci-fi robot",
    "rural landscape",
    "baroque",
    "intricate details",
    "minimalist landscape",
    "clear sky",
    "cute fantasy",
    "soft glow",
    "urban noir",
    "pet photography",
    "happy",
    "chinese traditional",
    "aged texture",
    "calligraphy",
    "antique",
    "volcanic fantasy",
    "fiery red",
    "rugged stone",
    "minimalist fashion",
    "natural light",
    "japanese myth",
    "feather cloak",
    "dessert photography",
    "delicate",
    "sweet",
    "futuristic architecture",
    "glass reflection",
    "bright light",
    "cursed glow",
    "graveyard",
    "spooky",
    "soft mist",
    "ink tone",
    "mid century modern",
    "wood texture",
    "cosmic glow",
    "dark background",
    "rural american",
    "natural texture",
    "anime supernatural",
    "energy glow",
    "classical greek art",
    "museum light",
    "tropical nature",
    "mist",
    "vibrant green",
    "luxury decor",
    "luxury",
    "japanese minimalist",
    "fantasy combat",
    "fire glow",
    "magic circle",
    "lifestyle photography",
    "macro fantasy",
    "intricate texture",
    "architectural photography",
    "clean lines",
    "japanese horror",
    "eerie glow",
    "bakery photography",
    "inviting",
    "fantasy celestial",
    "marble texture",
    "starry sky"
]

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("Calculating CLIP embeddings for style terms...")
style_inputs = clip_processor(text=STYLE_VOCAB, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    style_embeddings = clip_model.get_text_features(**style_inputs)
style_embeddings = style_embeddings / style_embeddings.norm(dim=-1, keepdim=True)  # 归一化

# get Deepseek client
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)

def extract_style_keywords(image_path: str, top_k: int = random.randint(3, 7)) -> List[str]:

    img = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embedding = clip_model.get_image_features(**inputs)
    image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
    
    similarity = (image_embedding @ style_embeddings.T).squeeze(0)
    top_indices = similarity.topk(min(top_k, len(STYLE_VOCAB))).indices.tolist()
    return [STYLE_VOCAB[i] for i in top_indices]

def generate_prompt_keywords(style_keywords: List[str]) -> str:

    style_str = ", ".join(style_keywords)
    system_prompt = (
        "You are a keyword generator for Stable Diffusion prompts. "
        "Your output must be ONLY a comma-separated list of keywords. "
        "No sentences, no explanations, no extra punctuation, no line breaks. "
        "Format: subject_keywords, style_keywords."
    )
    user_message = (
        f"Style keywords: {style_str}\n"
        "Generate a concise comma-separated keyword list describing a specific Vincent van Gogh style painting. "
        "Include 1-3 core subject keywords first (e.g., starry night, sunflowers, wheat field, self portrait, cafe terrace), "
        "then append ALL the provided style keywords exactly as given. "
        "Do not add any other words. Do not write a sentence. Output only the comma-separated keywords."
    )
    
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=80,
                stream=False
            )
            prompt = response.choices[0].message.content.strip()
            if ". " in prompt or "\n" in prompt:
                prompt = prompt.split(".")[0].split("\n")[0]
            prompt = prompt.strip('"').strip("'").strip()
            prompt = prompt.rstrip(".")
            return prompt
        except Exception as e:
            print(f" API failed  (try {attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(2 ** attempt)
    return f"Van Gogh painting, {style_str}"

def process_images():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    results = []
    processed_images = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    processed_images.add(os.path.basename(item["image"]))
                    results.append(item)
                except:
                    pass
        print(f"An existing output file has been detected .Have processed {len(processed_images)} images")
    
    for idx in range(START_INDEX, END_INDEX + 1):
        img_filename = f"{idx:04d}.jpg"
        img_path = os.path.join(IMAGE_DIR, img_filename)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} does not exist, skipping.")
            continue
        
        if img_filename in processed_images:
            continue  # Already processed
        
        print(f"Processing {img_filename}...")

        keywords = extract_style_keywords(img_path, top_k=random.randint(3, 7))
        print(f"  Style keywords: {keywords}")
        
        prompt = generate_prompt_keywords(keywords)
        print(f"  Generated prompt: {prompt}")
        result = {"image": img_path, "prompt": prompt}
        results.append(result)

        if len(results) % BATCH_SIZE == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Progress saved, total {len(results)} records.")

        time.sleep(0.5)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"All done! Total {len(results)} records generated and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_images()