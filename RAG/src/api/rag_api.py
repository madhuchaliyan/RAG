from src.components.rag_model import RAGModel

rag_instance = RAGModel()

class RAGAPI:
    def send_prompt(self, prompt, max_tokens=50, temperature=0.7, **kwargs):
        try:
            result = rag_instance.invoke_rag(prompt)
            return result
        except Exception:
            return "Server error"
       
