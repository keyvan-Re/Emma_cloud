import os
import json
import numpy as np
import tiktoken
from tqdm import tqdm
from app import classify_query_local # یا classify_query_openai بسته به انتخابتان

# فرض می‌کنیم توابع بازیابی شما در ماژول‌های زیر هستند (نام‌ها را در صورت نیاز تطبیق دهید)
# from memory_utils import load_user_memory
# from build_memory_index import load_indices

def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """شمارش تعداد توکن‌های یک متن با استفاده از توکنایزر"""
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

def simulate_retrieval(query, query_category, indices, semantic_text, top_k=3):
    """
    شبیه‌سازی بازیابی بر اساس طبقه‌بندی (EMMA - Routed)
    """
    retrieved_text = ""
    
    if query_category == "semantic_memory":
        retrieved_text = semantic_text
        
    elif query_category == "episodic_memory":
        if indices.get("episodic_memory"):
            # بازیابی از ایندکس اپیزودیک
            nodes = indices["episodic_memory"].as_retriever(similarity_top_k=top_k).retrieve(query)
            retrieved_text = "\n".join([n.node.text for n in nodes])
            
    elif query_category == "semantic-episodic":
        retrieved_text = semantic_text + "\n"
        if indices.get("episodic_memory"):
            nodes = indices["episodic_memory"].as_retriever(similarity_top_k=top_k).retrieve(query)
            retrieved_text += "\n".join([n.node.text for n in nodes])
            
    # اگر unknown باشد، معمولاً فقط از کانتکست مکالمه (session) استفاده می‌شود
    return retrieved_text

def simulate_full_retrieval(query, indices, semantic_text, top_k=3):
    """
    شبیه‌سازی بازیابی از تمام حافظه‌ها (EMMA-FULL)
    """
    retrieved_text = semantic_text + "\n"
    if indices.get("episodic_memory"):
        nodes = indices["episodic_memory"].as_retriever(similarity_top_k=top_k).retrieve(query)
        retrieved_text += "\n".join([n.node.text for n in nodes])
    if indices.get("sessions_memory"):
        nodes = indices["sessions_memory"].as_retriever(similarity_top_k=top_k).retrieve(query)
        retrieved_text += "\n".join([n.node.text for n in nodes])
        
    return retrieved_text

def run_benchmark(test_queries, username="test_user"):
    # ۱. بارگذاری ایندکس‌های کاربر (باید توابع واقعی پروژه خود را فراخوانی کنید)
    # indices = {
    #    "episodic_memory": load_index(username, "episodic"),
    #    "sessions_memory": load_index(username, "sessions")
    # }
    # semantic_text = get_semantic_memory(username)
    
    # مقادیر دامی برای جلوگیری از خطا در صورت عدم اتصال به دیتابیس در این مثال
    indices = {} 
    semantic_text = "User values empathy and stability. Big Five: High Openness." 
    
    results = {
        "EMMA": [],       # توکن‌های بازیابی شده با کلاسیفایر
        "EMMA_FULL": []   # توکن‌های بازیابی شده با جستجوی سراسری
    }
    
    for query in tqdm(test_queries, desc="Benchmarking Queries"):
        # ۲. طبقه‌بندی کوئری
        category = classify_query_local(query) 
        
        # ۳. بازیابی هدفمند (رویکرد پیشنهادی شما)
        routed_context = simulate_retrieval(query, category, indices, semantic_text)
        tokens_routed = count_tokens(routed_context)
        
        # ۴. بازیابی کامل (بیس‌لاین)
        full_context = simulate_full_retrieval(query, indices, semantic_text)
        tokens_full = count_tokens(full_context)
        
        results["EMMA"].append(tokens_routed)
        results["EMMA_FULL"].append(tokens_full)

    # ۵. محاسبه آمار و ارقام برای مقاله
    avg_emma = np.mean(results["EMMA"])
    avg_full = np.mean(results["EMMA_FULL"])
    reduction_pct = ((avg_full - avg_emma) / avg_full) * 100 if avg_full > 0 else 0

    print("\n" + "="*40)
    print(" TOKEN EFFICIENCY ANALYSIS RESULTS ")
    print("="*40)
    print(f"Total Queries Evaluated: {len(test_queries)}")
    print(f"Average Tokens Retrieved (EMMA-FULL): {avg_full:.2f}")
    print(f"Average Tokens Retrieved (EMMA Routed): {avg_emma:.2f}")
    print(f"Token Reduction (R_token): {reduction_pct:.2f}%")
    print("="*40)

if __name__ == "__main__":
    # مجموعه‌ای از کوئری‌های تستی برای ارزیابی
    sample_queries = [
        "What did we talk about regarding my job stress last week?", # Expected: episodic
        "Can you remind me what my core values are?",                # Expected: semantic
        "I'm feeling very overwhelmed today, just like last time.",  # Expected: semantic-episodic
        "Hi, how are you?"                                           # Expected: unknown
    ]
    
    run_benchmark(sample_queries)
