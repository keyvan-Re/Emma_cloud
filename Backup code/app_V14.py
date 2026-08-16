# -*- coding:utf-8 -*-

import os
import sys
import json
import time
import copy
import shutil
import signal
import logging
import platform
import re

import gradio as gr
import nltk
import torch
import tiktoken

from openai import OpenAI as OpenAIClient
from llama_index.embeddings.openai import OpenAIEmbedding

# from langchain_openai import ChatOpenAI, OpenAI as LangChainOpenAI # (If not used, can be commented out)
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.core import Settings
from llama_index.core.indices.prompt_helper import PromptHelper

prompt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(prompt_path)

# Assuming these imports exist in your local environment
from utils.sys_args import data_args, model_args
from utils.app_modules.utils import *
from utils.app_modules.presets import *
from utils.app_modules.overwrites import *
from utils.prompt_utils import *
from utils.memory_utils import (
    enter_name_llamaindex,
    summarize_memory_event_personality,
    save_local_memory,
    extract_session_summary,
    extract_semantic_memory,
)

# Ensure NLTK data path
nltk.data.path = [os.path.join(os.path.dirname(__file__), "nltk_data")] + nltk.data.path

tokenizer = tiktoken.get_encoding("cl100k_base")

GAPGPT_BASE_URL = os.getenv("GAPGPT_BASE_URL", "https://api.gapgpt.app/v1")
openai_client_cache = {}

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


LOCAL_MODEL_PATH = r"C:\Users\keyva\MMPL_gpt\Classification Model\final_xlm_r_model_router"

print(f"Loading model from: {LOCAL_MODEL_PATH}")

try:
    _tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    _classifier_model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)
    _classifier_model.eval()
    print("Local classifier loaded successfully.")
except Exception as e:
    print(f"Error loading local classifier: {e}")


def get_gapgpt_client(api_key: str) -> OpenAIClient:
    if not api_key:
        raise ValueError("API key is missing while attempting to create a GapGPT client.")
    if api_key not in openai_client_cache:
        openai_client_cache[api_key] = OpenAIClient(api_key=api_key, base_url=GAPGPT_BASE_URL)
    return openai_client_cache[api_key]


os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def signal_handler(signal_number, frame):
    global stop_stream
    stop_stream = True


VECTOR_SEARCH_TOP_K = 2

# Update this path to your actual file location
api_path = 'C:\\Users\\keyva\\MMPL_gpt\\api_key_list.txt'


def read_apis(path):
    api_keys_local = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if line:
                    api_keys_local.append(line)
    return api_keys_local


memory_dir = os.path.expanduser("C:\\Users\\keyva\\MMPL_gpt\\memories\\update_memory_0512_eng.json")

# Ensure directory exists
os.makedirs(os.path.dirname(memory_dir), exist_ok=True)

if not os.path.exists(memory_dir):
    json.dump({}, open(memory_dir, "w", encoding="utf-8"))

global memory
memory = json.load(open(memory_dir, "r", encoding="utf-8"))
language = 'en'
user_keyword = generate_user_keyword()[language]
ai_keyword = generate_ai_keyword()[language]
boot_name = boot_name_dict[language]
boot_actual_name = boot_actual_name_dict[language]
meta_prompt = generate_meta_prompt_dict_chatgpt()[language]
meta_prompt_semantic = generate_meta_prompt_dict_semantic_chatgpt()[language]
meta_prompt_semantic_episodic = generate_meta_prompt_dict_semantic_episodic_chatgpt()[language]
new_user_meta_prompt = generate_new_user_meta_prompt_dict_chatgpt()[language]

api_keys = read_apis(api_path)
# Fallback if file is empty or missing for testing purposes
if not api_keys:
    print(f"Warning: No API keys found in {api_path}.")

new_conversation = False
chatgpt_config = {
    "model": "gpt-4o",
    "temperature": 1,
    "max_tokens": 1024,
    "top_p": 0.95,
    "frequency_penalty": 0.4,
    "presence_penalty": 0.2,
    "n": 1,
}

deactivated_keys = []
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
)


def chatgpt_chat(prompt, system, history, gpt_config, api_index=0):
    """
    Handles the chat request to OpenAI.
    History is expected to be a list of dictionaries: [{'role': 'user', 'content': '...'}, ...]
    """
    retry_times, count = 5, 0
    response = None

    while response is None and count < retry_times:
        try:
            request = copy.deepcopy(gpt_config)
            
            # Initial system and greeting messages
            if data_args.language == 'en':
                message = [
                    {"role": "system", "content": system.strip()},
                    {"role": "user", "content": "Hi!"},
                    {"role": "assistant",
                     "content": f"Hi! I'm {boot_actual_name}! I will give you warm companion!"},
                ]
            else:
                message = [
                    {"role": "system", "content": system.strip()},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant",
                     "content": f"Hi! I'm {boot_actual_name}! I will give you warm companion!"},
                ]

            # --- FIX: Handle History as List of Dicts ---
            if history:
                # Verify if history is old format (list of lists) or new (list of dicts)
                if isinstance(history[0], list):
                     # Convert old format temporarily if encountered
                     for q, a in history:
                         message.append({"role": "user", "content": str(q)})
                         message.append({"role": "assistant", "content": str(a)})
                elif isinstance(history[0], dict):
                    # New format
                    for msg in history:
                        if msg.get('role') in ['user', 'assistant']:
                            message.append(msg)

            # Add the current prompt
            message.append({"role": "user", "content": f"{prompt}"})

            if not api_keys:
                return "Error: No API Keys available."
                
            client = get_gapgpt_client(api_keys[api_index])
            
            # New OpenAI 1.x Syntax
            response = client.chat.completions.create(messages=message, **request)

        except Exception as e:
            print(f"Chat Error: {e}")
            if 'This key is associated with a deactivated account' in str(e):
                deactivated_keys.append(api_keys[api_index])

            if api_keys:
                api_index = api_index + 1 if api_index < len(api_keys) - 1 else 0
                loop_check = 0
                while api_keys[api_index] in deactivated_keys and loop_check < len(api_keys):
                    api_index = api_index + 1 if api_index < len(api_keys) - 1 else 0
                    loop_check += 1
            count += 1

    if response:
        response = response.choices[0].message.content
    else:
        response = ''
    return response


def classify_query_local(text):
  
    
    # 0: episodic, 1: semantic, 2: semantic_episodic, 3: unrelated
    id2label_map = {
        0: "episodic_memory",
        1: "semantic_memory",
        2: "semantic-episodic",  
        3: "unknown"             
    }

    try:
       
        inputs = _tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        
       
        with torch.no_grad():
            outputs = _classifier_model(**inputs)
        
        
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        
        
        category = id2label_map.get(predicted_class_id, "unknown")
        
        print(f"Local Classifier: '{text}' -> {category} (Class ID: {predicted_class_id})")
        return category

    except Exception as e:
        print(f"Error in local classification: {e}")
        return "unknown"



def classify_query_openai(text, gpt_config, api_index=0, retry_times=5):
    response = None
    count = 0
    local_deactivated = []
    print("text********:", text)

    system_prompt = """
You are an AI that classifies user queries into one of the following memory types:
- 'episodic_memory': Queries about past personal events, daily life logs, or specific experiences the user has shared (e.g., "What did I eat yesterday?", "Tell me about my trip").
- 'semantic_memory': Queries about facts, preferences, general knowledge the user has taught you, or summaries of their personality (e.g., "What is my favorite color?", "Do I like sci-fi movies?").
- 'semantic-episodic': Complex queries requiring both specific past events and general facts/preferences (e.g., "Based on my food preferences, did I enjoy the dinner last night?").

Output ONLY one of these three strings: 'episodic_memory', 'semantic_memory', or 'semantic-episodic'. If unsure, output 'episodic_memory'.
    """.strip()
    
    if not api_keys:
        return "unknown"

    while response is None and count < retry_times:
        try:
            client = get_gapgpt_client(api_keys[api_index])
            
            # New OpenAI 1.x Syntax
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text.strip()},
                ],
                **copy.deepcopy(gpt_config),
            )
        except Exception as e:
            print(f"Classify Error: {e}")
            if "This key is associated with a deactivated account" in str(e):
                local_deactivated.append(api_keys[api_index])

            api_index = api_index + 1 if api_index < len(api_keys) - 1 else 0
            while api_keys[api_index] in local_deactivated:
                api_index = api_index + 1 if api_index < len(api_keys) - 1 else 0
            count += 1

    if response:
        category = response.choices[0].message.content.strip().lower()
        # Basic cleanup
        if "semantic_memory" in category: category = "semantic_memory"
        elif "semantic-episodic" in category: category = "semantic-episodic"
        elif "episodic_memory" in category: category = "episodic_memory"
        else: category = "unknown"
    else:
        category = "unknown"

    return category


def predict_new(
    text,
    history,
    top_p,
    temperature,
    max_length_tokens,
    max_context_length_tokens,
    user_name,
    user_memory,
    user_memory_index,
    service_context,
    api_index,
    semantic_memory_text,
    query_category,
    user_profile=None
     

):
    chatgpt_cfg = {
        "model": "gpt-4o",
        "temperature": temperature,
        "max_tokens": max_length_tokens,
        "top_p": top_p,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.2,
        "n": 1,
    }

    if text == "":
        # Return same history if empty input
        return history, history, "Empty context."
        
    # Ensure history is initialized
    if history is None:
        history = []

    system_prompt, related_memos = build_prompt_with_search_memory_llamaindex(
        history=history,
        query=text, # Changed name to match prompt_utils
        user_memory=user_memory,
        user_name=user_name,
        user_memory_index=user_memory_index,
        service_context=service_context,
        api_keys=api_keys,
        api_index=api_index,
        meta_prompt=meta_prompt,
        new_user_meta_prompt=new_user_meta_prompt,
        data_args=data_args,
        boot_actual_name=boot_actual_name,
        semantic_memory_text=semantic_memory_text,
        query_category=query_category,
        meta_prompt_semantic=meta_prompt_semantic,
        meta_prompt_semantic_episodic=meta_prompt_semantic_episodic,
    )

    # ==========================================
    # اضافه کردن پروفایل به System Prompt
    # ==========================================
    if user_profile:
        profile_str = (
            f"\n\n[User Profile Information]:\n"
            f"- Age: {user_profile.get('age', 'Unknown')}\n"
            f"- Gender: {user_profile.get('gender', 'Unknown')}\n"
            f"- Occupation: {user_profile.get('occupation', 'Unknown')}\n"
            f"- Residence: {user_profile.get('residence', 'Unknown')}\n"
            f"Use this context subtly when talking to the user."
        )
        system_prompt += profile_str
    # ==========================================

    # Handle context window slicing manually if needed, 
    # though usually OpenAI manages this, or we slice the list of dicts.
    current_history_for_llm = history
    if len(history) > data_args.max_history * 2: # *2 because 1 user + 1 bot
        current_history_for_llm = history[-(data_args.max_history * 2):]

    response = chatgpt_chat(
        prompt=text,
        system=system_prompt,
        history=current_history_for_llm,
        gpt_config=chatgpt_cfg,
        api_index=api_index,
    )

    torch.cuda.empty_cache()

    # --- FIX: Update History with Dictionaries (Gradio Chatbot format) ---
    new_history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": response}
    ]

    # Save memory logic (Needs to handle dict format, assuming save_local_memory can handle it 
    # OR we convert strictly for saving if your legacy code needs it. 
    # For now, assuming we pass the object as is or convert for save)
    if user_name:
        # If save_local_memory expects list of lists, we might need to adapt it inside that function
        # or pass a converted version. Let's try passing the new format.
        save_local_memory(memory, new_history, user_name, data_args)

    # Return: (Chatbot View, State History, Textbox Reset)
    return new_history, new_history, "Generating..."



def create_gradio_interface(service_context, api_keys):
    

    # ابتدا متغیر CSS را تعریف کنید
    custom_css = """
/* استایل دکمه ارسال (Send) */
.send-btn {
    border: 3px solid #0066cc !important;
    border-radius: 12px !important;
    background-color: white !important;
    color: black !important;
    font-weight: bold !important;
    font-size: 16px !important;
    transition: all 0.3s ease !important;
}
.send-btn:hover {
    background-color: #0066cc !important;
    color: white !important;
}

/* استایل نوار پایین و دکمه‌های آن */
.bottom-actions-row {
    background-color: #e5e7eb !important; /* رنگ خاکستری پس‌زمینه */
    border-radius: 8px !important;
    padding: 5px !important;
    margin-top: 10px !important;
}

.action-btn {
    border: none !important;
    background: transparent !important;
    color: black !important;
    font-weight: bold !important;
    box-shadow: none !important;
    transition: all 0.3s ease !important;
}

.action-btn:hover {
    background-color: #d1d5db !important; /* تغییر رنگ ملایم هنگام هاور */
    border-radius: 8px !important;
}
/* ========================================== */
/* لودینگ تمام‌صفحه و غیرفعال‌سازی تعاملات */
/* ========================================== */

/* مخفی کردن نوار پیشرفت پیش‌فرض Gradio */
.progress-level, .progress-text, .progress-level svg, .progress-level img {
    display: none !important;
}

/* وقتی کلاسی به نام progress-level در صفحه ایجاد شود، این استایل‌ها روی body اعمال می‌شود */
body:has(.progress-level) {
    pointer-events: none !important; /* غیرفعال کردن تمام کلیک‌ها */
}

/* ایجاد هاله نیمه‌شفاف روی کل صفحه */
body:has(.progress-level)::after {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    background-color: rgba(255, 255, 255, 0.7); /* رنگ سفید نیمه‌شفاف */
    z-index: 9998;
    pointer-events: all; /* برای گرفتن کلیک‌ها و جلوگیری از عبور آن‌ها */
    cursor: wait;
}

/* ایجاد اسپینر لودینگ در مرکز صفحه */
body:has(.progress-level)::before {
    content: "";
    position: fixed;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 60px; height: 60px;
    border: 6px solid #e5e7eb;
    border-top-color: #0066cc;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    z-index: 9999;
}

@keyframes spin {
    0% { transform: translate(-50%, -50%) rotate(0deg); }
    100% { transform: translate(-50%, -50%) rotate(360deg); }
}


/* انیمیشن چرخش */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* تغییر کلمه Processing (اختیاری) - مخفی کردن متن پیش فرض */
/* اگر می‌خواهید کلا متنی نباشد و فقط دایره بچرخد، کد زیر را از کامنت در بیاورید */
/*
.progress-text {
    display: none !important;
}
*/
/* انیمیشن سه نقطه در حال تایپ */
.typing-dots::after {
    content: '';
    animation: typing 1.5s infinite;
}
@keyframes typing {
    0% { content: ''; }
    25% { content: ' .'; }
    50% { content: ' . .'; }
    75% { content: ' . . .'; }
    100% { content: ''; }
}


"""

    with gr.Blocks(title="EMMA") as demo:
        # حالا متغیر را به HTML پاس دهید (بدون علامت مساوی)
        gr.HTML(f"<style>{custom_css}</style>")
        
    # بقیه کدهای شما...
        
        
        state = gr.State({
            "history": [], 
            "user_name": None,
            "memory": memory, # فرض بر این است که متغیر memory خارج از تابع تعریف شده
            "data_args": data_args, # فرض بر این است که data_args خارج از تابع تعریف شده
            "service_context": service_context,
            "api_keys": api_keys,
            "api_index": 0,
            "semantic_memory_text": "",
            "new_conversation": True,
            "initialized": False
        })

        

        # فرض بر این است که این کد داخل with gr.Blocks() قرار دارد
        with gr.Row():
            
                                    # ==========================================
                        # 1. LOGIN PAGE (سایدبار سمت چپ)
            # ==========================================
            # استفاده از gr.Sidebar به جای gr.Column و gr.Accordion
            with gr.Sidebar(open=True) as login_page:
                
                gr.Markdown("<h1 style='text-align: center;'>🧠 EMMA</h1><h3 style='text-align: center;'>Your Empathetic Mental Health Assistant</h3><br><p style='text-align: center;'>Please login or register to begin.</p>")
                
                username_input = gr.Textbox(label="Your Name (Required)", placeholder="e.g., Alex")
                login_status = gr.Textbox(visible=False, label="Status")
                
                check_user_btn = gr.Button("🔍 Login / Check", variant="primary", elem_classes=["login-btn"])
                
                with gr.Column(visible=False) as registration_fields:
                    gr.Markdown("### ✨ New User Registration")
                    age_input = gr.Textbox(label="Age", placeholder="e.g., 28")
                    gender_input = gr.Dropdown(label="Gender", choices=["Male", "Female", "Other"])
                    occupation_input = gr.Textbox(label="Occupation", placeholder="e.g., Student...")
                    residence_input = gr.Textbox(label="Place of Residence", placeholder="e.g., Berlin")
                    register_btn = gr.Button("🎯 Complete Registration", variant="primary", elem_classes=["login-btn"])


            # ==========================================
            # 2. CHAT PAGE (ستون سمت راست)
            # ==========================================
            # مقدار visible=False حذف شده و scale=3 اضافه شده است تا بخش چت بزرگتر باشد
            with gr.Column(scale=3, elem_classes=["chat-box"]) as chat_page:
                active_header = gr.Markdown("<h2 style='text-align: center; color: #333;'>🧠 EMMA: Session</h2>")
                #system_msg = gr.Textbox(label="🔔 System Messages", interactive=False, max_lines=2)

                with gr.Group():
                    chatbot = gr.Chatbot(label="💬 EMMA Conversation", height=350)

                    with gr.Row():
                        user_input = gr.Textbox(placeholder="Type your message here...", show_label=False, scale=6)
                        # اختصاص کلاس send-btn به دکمه Send
                        submit_btn = gr.Button("📤 Send", elem_classes=["send-btn"], scale=1)

                    # اختصاص کلاس bottom-actions-row به ردیف دکمه‌های پایین
                    with gr.Row(equal_height=True, elem_classes=["bottom-actions-row"]):
                        # اختصاص کلاس action-btn به دکمه‌های عملیاتی
                        clear_btn = gr.Button("🧹 Clear", elem_classes=["action-btn"])
                        new_session_btn = gr.Button("🔄 New Session", elem_classes=["action-btn"])
                        switch_user_btn = gr.Button("👥 Logout", elem_classes=["action-btn"])
                system_msg = gr.Textbox(label="🔔 System Messages", interactive=False, max_lines=2)

        # -------------------------------------------------------
        # Internal Functions
        # -------------------------------------------------------
        def handle_login_check(name, state):
            if not name.strip():
                return (gr.update(), gr.update(), gr.update(), gr.update(visible=True, value="⚠️ Please enter a valid name."), gr.update(), gr.update(), state)
            
            if name in memory:
                hello_msg, user_memory, sessions_memory, episodic_memory, semantic_memory = enter_name_llamaindex(name, memory, data_args)
                user_memory = summarize_memory_event_personality(data_args, memory, name)
                
                new_state = state.copy()
                new_state["user_name"] = name
                new_state["memory"] = memory
                new_state["initialized"] = True
                new_state["semantic_memory_text"] = user_memory.get("semantic_memory", {})
                new_state["semantic_memory_index"] = semantic_memory
                new_state["user_memory_index"] = sessions_memory # یا هر نامی که ایندکس در آن است
                new_state["episodic_memory_index"] = episodic_memory
                new_state["history"] = []
                


                
                welcome_msg = hello_msg if hello_msg else f"Welcome back, {name}!"
                
                return (
                    gr.update(open=False), # 2. افزودن کلاس انیمیشن به جای visible=False
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False, value=""),
                    gr.update(value=f"<h2 style='text-align: center; color: #333;'>🧠 EMMA: Session for {name}</h2>"),
                    gr.update(value=welcome_msg),
                    new_state
                )
            else:
                return (
                    gr.update(), # تغییری در صفحه اصلی لاگین ایجاد نمیشود
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True, value="📝 New user detected. Please fill the details below."), 
                    gr.update(),
                    gr.update(),
                    state
                )

        def handle_register(name, age, gender, occupation, residence, state):
            name = str(name).strip()

            if not name:
                raise gr.Error("Please enter a valid user name.")

            # Registration is only for new users.
            if name in memory:
                raise gr.Error(
                    f"User '{name}' already exists. Please use the login form."
                )

            # Create the user before trying to build or load their indices.
            memory[name] = {
                "profile": {
                    "age": age,
                    "gender": gender,
                    "occupation": occupation,
                    "residence": residence,
                },
                "sessions": [],
                "episodic_memory": [],
                "semantic_memory": {},
            }

            try:
                (
                    _,
                    user_memory,
                    sessions_memory,
                    episodic_memory,
                    semantic_memory,
                ) = enter_name_llamaindex(
                    name=name,
                    memory=memory,
                    data_args=data_args,
                    update_memory_index=True,
                )
            except Exception:
                # Avoid leaving a partially registered user in memory.
                memory.pop(name, None)
                raise

            new_state = dict(state or {})
            new_state.update(
                {
                    "user_name": name,
                    "memory": memory,
                    "initialized": True,
                    "history": [],
                    "semantic_memory_text": user_memory.get("semantic_memory", {}),
                    "user_memory_index": sessions_memory,
                    "episodic_memory_index": episodic_memory,
                    "semantic_memory_index": semantic_memory,
                }
            )

            welcome_msg = f"Welcome {name}! Registration complete."

            return (
                gr.update(open=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False, value=""),
                gr.update(
                    value=(
                        "<h2 style='text-align: center; color: #333;'>"
                        f"🧠 EMMA: Session for {name}</h2>"
                    )
                ),
                gr.update(value=welcome_msg),
                new_state,
    )

        def switch_user(state):
            if state["initialized"] and state["user_name"] in state["memory"]:
                if state["memory"][state["user_name"]]["sessions"]:
                    previous_session = state["memory"][state["user_name"]]["sessions"][-1]
                    summary = extract_session_summary(previous_session["conversation"], previous_session["date"], len(state["memory"][state["user_name"]]["sessions"]) - 1)
                    state["memory"][state["user_name"]]["episodic_memory"].append(summary)
                    # ==========================================
                    # به‌روزرسانی حافظه سمانتیک زمان خروج
                    # ==========================================
                    try:
                        user_mem = state["memory"][state["user_name"]]
                        existing_semantic = user_mem.get("semantic_memory", {})
                        
                        updated_semantic = extract_semantic_memory(summary, existing_semantic)
                        
                        state["memory"][state["user_name"]]["semantic_memory"] = updated_semantic
                    except Exception as e:
                        print(f"Error updating semantic memory on logout: {e}")
                    # ==========================================

            new_state = state.copy()
            new_state.update({"history": [], "user_name": None, "semantic_memory_text": "", "initialized": False})

            return (
                gr.update(open=True),                   # 1. باز شدن سایدبار
                gr.update(visible=False),               # 2. بسته شدن فرم ثبت‌نام
                gr.update(visible=False),               # 3. مخفی شدن صفحه چت
                gr.update(value="Logged out successfully.", visible=True), # 4. پیام خروج
                gr.update(value=""),                    # 5. پاک کردن نام کاربری
                new_state,                              # 6. استیت جدید
                gr.update(value=[]),                    # 7. پاک کردن چت‌بات
                gr.update(value=""),                    # 8. پاک کردن age_input
                gr.update(value=""),                    # 9. پاک کردن gender_input
                gr.update(value=""),                    # 10. پاک کردن occupation_input
                gr.update(value=""),                    # 11. پاک کردن residence_input
                gr.update(value="")                     # 12. پاک کردن پیام سیستم (system_msg) اضافه شد
            )


        def handle_chat(user_message, state):
            if not user_message.strip():
                print("[handle_chat] empty message")
                print("[handle_chat] state =", state)
                # تبدیل return به yield
                yield gr.update(), state.get("history", []), state
                return

            print("[handle_chat] user_message =", repr(user_message))
            print("[handle_chat] state keys =", list(state.keys()) if isinstance(state, dict) else type(state))
            print("[handle_chat] user_name =", state.get("user_name"))
            print("[handle_chat] user_memory_index =", state.get("user_memory_index"))
            print("[handle_chat] memory exists =", state.get("memory") is not None)
            print("[handle_chat] service_context exists =", state.get("service_context") is not None)
            print("[handle_chat] api_index =", state.get("api_index", 0))
            print("[handle_chat] semantic_memory_text =", repr(state.get("semantic_memory_text", "")))
            print("[handle_chat] history len =", len(state.get("history", [])))

            # ==========================================
            # ۱. نمایش فوری پیام کاربر و انیمیشن لودینگ
            # ==========================================
            current_history = state.get("history", [])
            temp_history = current_history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "<span class='typing-dots'>Generating</span>"}
            ]
            
            # باکس متنی را پاک کرده و وضعیت موقت را بلافاصله به UI می‌فرستیم
            yield gr.update(value=""), temp_history, state


            # ادامه پردازش‌های اصلی تابع شما
            user_name = state.get("user_name")
            user_memory_index = state.get("user_memory_index")
            user_memory = state.get("user_memory", {})
            user_memory_index = state.get("user_memory_index")
            semantic_memory_text = state.get("semantic_memory_text", {})
            service_context = state.get("service_context")
            api_index = state.get("api_index", 0)
            semantic_memory_text = state.get("semantic_memory_text", "")

            # ==========================================
            # استخراج پروفایل کاربر از حافظه
            # ==========================================
            user_profile = {}
            if user_name and state.get("memory") and user_name in state["memory"]:
                user_profile = state["memory"][user_name].get("profile", {})
            # ==========================================    

            query_category = classify_query_local(user_message)
            print("[handle_chat] query_category =", query_category)

            # ارسال current_history به مدل (بدون انیمیشن موقت)
            new_history, _, status_msg = predict_new(
                text=user_message,
                history=current_history,
                top_p=0.95,
                temperature=1.0,
                max_length_tokens=1024,
                max_context_length_tokens=4096,
                user_name=user_name,
                user_memory=user_memory,
                user_memory_index=user_memory_index,
                service_context=service_context,
                api_index=api_index,
                semantic_memory_text=semantic_memory_text,
                query_category=query_category,
                user_profile=user_profile 
            )

            print("[handle_chat] status_msg =", status_msg)
            print("[handle_chat] new_history len =", len(new_history))

            state["history"] = new_history
            print("[handle_chat] updated state history len =", len(state.get("history", [])))

            # ==========================================
            # ۳. جایگزینی پیام لودینگ با پاسخ نهایی
            # ==========================================
            yield gr.update(), new_history, state
                    
        def clear_history(state):
            state["history"] = []
            return [], state
        
        def handle_new_session(state):
            # بررسی اینکه آیا کاربر لاگین کرده است یا خیر
            user_name = state.get("user_name")
            if not user_name or "memory" not in state or user_name not in state["memory"]:
                print("Error: User not logged in or memory missing.")
                # خروجی اول برای chatbot (خالی) و خروجی دوم برای state
                return [], state, gr.update() # خروجی سوم اضافه شد

            # دریافت اطلاعات کاربر فعلی از داخل حافظه کلی
            user_data = state["memory"][user_name]
            
            # --- بخش اول: استخراج حافظه از جلسه قبلی ---
            if user_data.get("sessions") and len(user_data["sessions"]) > 0:
                last_session = user_data["sessions"][-1]
                
                # فقط در صورتی حافظه را استخراج کن که جلسه قبلی خالی نباشد
                if len(last_session.get("conversation", [])) > 0:
                    session_date = last_session.get("date", "")
                    session_id = last_session.get("session_id", 0)
                    
                    # پاس دادن هر ۳ آرگومان به تابع
                    ep_summary = extract_session_summary(last_session["conversation"], session_date, session_id)
                    
                    if "episodic_memory" not in user_data:
                        user_data["episodic_memory"] = []
                    user_data["episodic_memory"].append(ep_summary)

                    # ==========================================
                    # به‌روزرسانی حافظه سمانتیک
                    # ==========================================
                    try:
                        # دریافت حافظه سمانتیک قبلی کاربر (اگر نداشت، یک دیکشنری خالی می‌دهیم)
                        existing_semantic = user_data.get("semantic_memory", {})
                        
                        # فراخوانی تابع با هر دو آرگومان
                        updated_semantic = extract_semantic_memory(ep_summary, existing_semantic)
                        
                        # ذخیره دیتای جدید
                        user_data["semantic_memory"] = updated_semantic 
                    except Exception as e:
                        print(f"Error updating semantic memory: {e}")
                    # ==========================================

            # --- بخش دوم: ایجاد جلسه جدید با اختصاص ID ---
            new_session_id = len(user_data.get("sessions", []))
            
            new_session = {
                "session_id": new_session_id,
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "conversation": []
            }
            
            if "sessions" not in user_data:
                user_data["sessions"] = []
                
            user_data["sessions"].append(new_session)
            
            # پاک کردن تاریخچه چت در state
            state["history"] = []
            
            # محاسبه شماره جلسه فعلی (چون از 0 شروع میشود، +1 میکنیم)
            session_number = len(user_data["sessions"])
            new_header_text = f"<h2 style='text-align: center; color: #333;'>🧠 EMMA: Session {session_number} for {user_name}</h2>"
            
            # بازگرداندن خروجی جدید برای active_header
            return [], state, gr.update(value=new_header_text)

        check_user_btn.click(
            handle_login_check,
            inputs=[username_input, state],
            outputs=[login_page, registration_fields, chat_page, login_status, active_header, system_msg, state],
            # show_progress="hidden"
        )

        register_btn.click(
            handle_register,
            inputs=[username_input, age_input, gender_input, occupation_input, residence_input, state],
            outputs=[login_page, registration_fields, chat_page, login_status, active_header, system_msg, state],
            # show_progress="hidden"
        )

        switch_user_btn.click(
            switch_user,
            inputs=[state],
            # system_msg به انتهای لیست خروجی‌ها اضافه شد
            outputs=[login_page, registration_fields, chat_page, login_status, username_input, state, chatbot, age_input, gender_input, occupation_input, residence_input, system_msg],
            # show_progress="hidden"
        )

        
        submit_btn.click(
            handle_chat,
            inputs=[user_input, state],
            outputs=[user_input, chatbot, state],
            # show_progress="hidden"
        )
        
        user_input.submit(
            handle_chat,
            inputs=[user_input, state],
            outputs=[user_input, chatbot, state],
            # show_progress="hidden"
        )
        
        clear_btn.click(
            clear_history,
            inputs=[state],
            outputs=[chatbot, state],
            # show_progress="hidden"
        )
        
                # تغییر از clear_history به handle_new_session
        new_session_btn.click(
            handle_new_session,
            inputs=[state],
            outputs=[chatbot, state, active_header], # active_header به خروجی اضافه شد
            # show_progress="hidden"
        )

    return demo


def main():
    """Main function to initialize and launch the interface."""
    gapgpt_api_key = os.getenv("GAPGPT_API_KEY")
    
    if not gapgpt_api_key:
        print("Warning: GAPGPT_API_KEY environment variable is not set. Proceeding with keys from file.")
    else:
        # تغییر مهم ۱: ست کردن متغیر محیطی برای جلوگیری از خطاهای ناگهانی کتابخانه‌های وابسته
        os.environ["OPENAI_API_KEY"] = gapgpt_api_key

    # تنظیم LLM (مدل زبانی)
    llm = LlamaIndexOpenAI(
        model="gpt-4o",
        temperature=1,
        max_tokens=1024,
        top_p=0.95,
        frequency_penalty=0.4,
        presence_penalty=0.2,
        api_key=gapgpt_api_key,
        api_base=GAPGPT_BASE_URL,
    )

    # تغییر مهم ۲: تنظیم مدل Embedding با همان کلید و آدرس GapGPT
    # اگر سرویس دهنده شما از امبدینگ پشتیبانی نمی‌کند، این بخش نیاز به تغییر به مدل لوکال دارد
    embed_model = OpenAIEmbedding(
        api_key=gapgpt_api_key,
        api_base=GAPGPT_BASE_URL,
        model="text-embedding-3-small" # یا هر مدلی که سرویس شما پشتیبانی می‌کند
    )

    # اعمال تنظیمات سراسری
    Settings.llm = llm
    Settings.embed_model = embed_model  # <--- این خط جلوی خطای فعلی را می‌گیرد

    Settings.prompt_helper = PromptHelper(
        context_window=4096,
        num_output=256,
        chunk_overlap_ratio=20 / 4096,
        tokenizer=tokenizer,
    )

    # توجه: مطمئن شوید متغیر api_keys در اینجا تعریف شده باشد یا از args خوانده شود
    # اگر api_keys در کد شما تعریف نشده، احتمالاً باید آن را بارگذاری کنید یا اگر استفاده نمی‌شود حذف کنید.
    # فرض بر این است که api_keys قبلاً در کد شما تعریف شده است:
    if 'api_keys' not in locals():
        api_keys = {} # یا هر مقداری که کد شما انتظار دارد

    demo = create_gradio_interface(Settings, api_keys)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True
    )


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()
