from llm_agent import LLMAgent
from sd_pipeline import SDPipeline
from clip_image_rag import ClipImageRAG
from style_search import StyleExtractor


def main():

    llm = LLMAgent()
    sd = SDPipeline()
    rag = ClipImageRAG()
    styleE = StyleExtractor()

    while True:
        user_input = input("User: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        # =========================
        #  RAG （text → embedding → similar images）
        # =========================
        rag_results = rag.search_style(user_input)
        #This design can be understood as a preference setting. 
        #Users place their preferred images in the style_images folder, 
        #and the system will prioritize retrieving images similar to these,
        #thereby extracting the style features that the user likes.

        # （embedding → style words）
        # =========================
        style_text = styleE.extract(rag_results)

        print("\n[Extracted Style]:\n", style_text)



        # =========================
        # 4️⃣ LLM Prompt Engineering
        # =========================
        messages = [
            {
                        "role": "system",
                        "content": """
        You are a Stable Diffusion prompt engineer.

        Rules:
        - Only output ONE final prompt
        - No explanation
        - Use comma-separated phrases
        - Must include:
        - subject
        - style
        - color
        - lighting
        - composition
        - Style words are VERY IMPORTANT
        - Higher similarity examples must influence more
        """
                    },
                    {
                        "role": "user",
                        "content": f"""
        User Input:
        {user_input}

        Extracted Style Keywords:
        {style_text}
        Generate a concise, comma-separated prompt for Stable Diffusion based on the above information.
        Output tokens must be slower than 77.       
        """
            }
        ]

        # =========================
        # generate argument prompt
        # =========================
        enhanced_prompt = llm.generate(messages,max_new_tokens=77)
        if "assistant" in enhanced_prompt:
            enhanced_prompt = enhanced_prompt.split("assistant")[-1]

        enhanced_prompt = enhanced_prompt.strip()
        print("\n[Enhanced Prompt]:\n", enhanced_prompt)

        # =========================
        # generate image
        # =========================
        image_path = sd.generate(enhanced_prompt)

        print(f"\n[Image saved at]: {image_path}\n")


if __name__ == "__main__":
    main()