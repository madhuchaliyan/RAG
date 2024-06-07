import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available
from transformers import BitsAndBytesConfig
from huggingface_hub import login
from src.utils.helpers import retrieve_relevant_resources, load_config_values

# https://huggingface.co/settings/tokens

# Use console to login to huggingface
# huggingface-cli login

class RAGModel:
    def __init__(self):
        load_config_values()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        login(os.environ.get('HUGGINGFACE_TOKEN'))
        self.model, self.use_quantization_config = self.select_model()
        self.llm_model, self.tokenizer = self.initialize_model(self.model, self.use_quantization_config)
        self.sent_chunks_embeddings = pd.read_csv("sent_chunks_embeddings.csv")
        self.sent_chunks_embeddings["embedding"] = self.sent_chunks_embeddings["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        self.pages_and_chunks = self.sent_chunks_embeddings.to_dict(orient="records")
        self.embeddings = torch.tensor(np.array(self.sent_chunks_embeddings["embedding"].tolist()), dtype=torch.float32).to(self.device)

    def select_model(self):
        gpu_mem_gb = round((torch.cuda.get_device_properties(0).total_memory) / (2 ** 30))

        if gpu_mem_gb < 5:
            print(f"Your GPU memory available is {gpu_mem_gb}GB, which is not enough to run a Gemma LLM locally.")
        elif gpu_mem_gb <= 8:
            print(f"GPU memory: {gpu_mem_gb} | Gemma 2B with quantization can be used.")
            use_quantization_config = True
            model = "google/gemma-2b-it"
        elif gpu_mem_gb <= 18:
            print(f"GPU memory: {gpu_mem_gb} | Gemma 2B without quantization can be used.")
            use_quantization_config = False
            model = "google/gemma-2b-it"
        else:
            print(f"GPU memory: {gpu_mem_gb} | Gemma 7B can be used.")
            use_quantization_config = False
            model = "google/gemma-7b-it"
        return model, use_quantization_config

    def initialize_model(self, model, use_quantization_config):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        attn_implementation = "flash_attention_2" if (is_flash_attn_2_available() and (torch.cuda.get_device_capability(0)[0] >= 8)) else "sdpa"
        print(f"[INFO] Using attention implementation: {attn_implementation}")

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model)
        llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model,
                                                         torch_dtype=torch.float16,
                                                         quantization_config=quantization_config if use_quantization_config else None,
                                                         low_cpu_mem_usage=False,
                                                         attn_implementation=attn_implementation)
        if not use_quantization_config:
            llm_model.to("cuda")
        return llm_model, tokenizer


    def prompt_formatter(self, query, context_items):
        context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])

        base_prompt = f"""Consider the provided context carefully to formulate your response to the query. 
        Take your time to extract relevant information before providing your answer. Focus on clarity 
        and thoroughness in your response, ensuring it addresses the query comprehensively without 
        including the process of your reasoning..

        Now use the following context items to answer the user query:
        {context}

        User query: {query}
        Answer:"""

        return base_prompt

    def generate_response(self, query, context_items, temperature=0.7, max_new_tokens=512, format_answer_text=True):
        prompt = self.prompt_formatter(query=query, context_items=context_items)
        input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.llm_model.generate(**input_ids, temperature=temperature, do_sample=True, max_new_tokens=max_new_tokens)
        output_text = self.tokenizer.decode(outputs[0])

        if format_answer_text:
            output_text = output_text.replace(prompt, "").replace("<bos>", "").replace("<eos>", "").replace("Sure, here is the answer to the user query:\n\n", "")

        return output_text

    def invoke_rag(self, user_input):
        scores, indices = retrieve_relevant_resources(query=user_input, embeddings=self.embeddings)
        context_items = [self.pages_and_chunks[i] for i in indices]
        answer = self.generate_response(user_input, context_items)
        return answer

#################################################################

rag_model = RAGModel()
import random

query_list = [
    "How does integrating blockchain technology enhance security in IoT solutions, and what specific vulnerabilities does it address?",
    "Can you explain the process of integrating IoT devices with a blockchain network and how it ensures data integrity and immutability?",
    "How LoRaWAN works?",
    "How blockchain and IoT can help in food supply chain?",
    "Is the food chain a good use case for IoT and blockchain technology?",
    "Can you explain the importance of cloud computing to Industry 4.0?"
]
query = random.choice(query_list)
print(f"Query: {query}")

answer = rag_model.invoke_rag(query)


print(f"Answer:\n")
print(answer) 
