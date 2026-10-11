### COMPLETE THE CODE ###
## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
from policy_loader import load_policy_documents
from rag import run_rag

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ========= MODIFY THIS PATH TO YOUR REAL Qwen3‑0.6b FOLDER =========
MODEL_LOCAL_PATH = r"C:\Users\张钰涵\Desktop\Python\scSE_26_GEx3\models\qwen3-0.6b"
# ====================================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_LOCAL_PATH,
    local_files_only=True
)
# Qwen3 手动补全pad token
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_LOCAL_PATH,
    dtype=torch.bfloat16,   
    device_map="auto",
    local_files_only=True
)


def generate_answer(user_query: str, retrieved_chunks):
    """
    Generate answer using qwen3‑0.6b with retrieved context.
    If information is missing in context, must state it is not specified, NO hallucination.
    Fix meta‑device tensor RuntimeError.
    """
    context_text = "\n".join(retrieved_chunks)

    prompt = f"""You are a university IT support assistant.
Use ONLY the provided policy context to answer the user's question.
If the required information is NOT present in the context, clearly state that you do not know or the information is not specified.
DO NOT invent any facts, phone numbers or other details. Keep your answer concise.

Context:
{context_text}

User question: {user_query}
Assistant:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_part = full_output.split("Assistant:")[-1].strip()
    answer_part = answer_part.split("\n")[0]   # 截断，防止模型循环重复输出
    return answer_part


def main():
    documents = load_policy_documents()

    question_set = [
        "What exact phone number should I call for the University IT Service Desk?",
        "How do I connect to the university campus wireless network?",
        "What software is required to use the university campus VPN?",
        "What is the default password for new university student email accounts?",
        "Where can I submit a ticket for IT support issues?",
        "What is the email address for University IT Service Desk?",
        "What should I do if I forget my university unified identity password?",
        "What devices are permitted to connect to the campus network?",
        "What is the opening time of the university IT helpdesk office?",
        "Can I use campus VPN outside of university premises?"
    ]

    for question in question_set:
        retrieved_chunks = run_rag(question, documents)
        final_ans = generate_answer(question, retrieved_chunks)

        print(f"QUESTION: {question}")
        print(f"Answer: {final_ans}")
        print("-" * 70) 


if __name__ == "__main__":
    main()
