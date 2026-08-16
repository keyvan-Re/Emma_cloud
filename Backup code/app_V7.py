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
    with gr.Blocks(title="EMMA") as demo:
        
        # 1. اضافه شدن CSS انیمیشن کشویی (Slide Up)
        gr.HTML("""
        <style>
            .center-content {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }
            /* -- استایل‌های فرم ورود -- */
            .login-box {
                max-width: 400px !important;
                margin: 0 auto !important;
                width: 100% !important;
            }
            .login-btn, .login-btn:hover {
                background-color: #3498db !important;
                background-image: none !important;
                color: white !important;
                border: none !important;
            }
            /* -- استایل‌های محیط چت -- */
            .chat-box {
                max-width: 800px !important;
                margin: 0 auto !important;
                width: 100% !important;
            }
            /* -- دکمه‌های محیط چت (آبی رنگ) -- */
            .mobile-button, .mobile-button:hover {
                background-color: #3498db !important;
                background-image: none !important;
                color: white !important;
                border: none !important;
            }
            .gr-textbox textarea {
                font-size: 16px;
            }
            .gr-chatbot {
                font-size: 15px;
            }
            
            /* --- انیمیشن کشویی لاگین --- */
            .slide-up {
                animation: slideUpAnimation 0.8s ease-in-out forwards !important;
            }

            @keyframes slideUpAnimation {
                0% {
                    transform: translateY(0);
                    opacity: 1;
                    max-height: 1000px;
                }
                100% {
                    transform: translateY(-100%);
                    opacity: 0;
                    max-height: 0px;
                    overflow: hidden;
                    margin: 0 !important;
                    padding: 0 !important;
                    border: none !important;
                }
            }
        </style>
        """)

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

        # ==========================================
        # 1. LOGIN PAGE
        # ==========================================
        with gr.Column(visible=True, elem_classes=["center-content"]) as login_page:
            gr.Markdown("<h1 style='text-align: center;'>🧠 EMMA</h1><h3 style='text-align: center;'>Your Empathetic Mental Health Assistant</h3><br><p style='text-align: center;'>Please login or register to begin.</p>")
            
            with gr.Column(scale=1, min_width=300, elem_classes=["login-box"]):
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
        # 2. CHAT PAGE
        # ==========================================
        with gr.Column(visible=False, elem_classes=["center-content"]) as chat_page:
            with gr.Column(elem_classes=["chat-box"]):
                active_header = gr.Markdown("<h2 style='text-align: center; color: #333;'>🧠 EMMA: Session</h2>")
                system_msg = gr.Textbox(label="🔔 System Messages", interactive=False, max_lines=2)

                with gr.Group():
                    chatbot = gr.Chatbot(label="💬 EMMA Conversation", height=450)

                    with gr.Row():
                        user_input = gr.Textbox(placeholder="Type your message here...", show_label=False, scale=4)
                        submit_btn = gr.Button("📤 Send", elem_classes=["mobile-button"], scale=1)

                    with gr.Row(equal_height=True):
                        clear_btn = gr.Button("🧹 Clear", elem_classes=["mobile-button"])
                        new_session_btn = gr.Button("🔄 New Session", elem_classes=["mobile-button"])
                        switch_user_btn = gr.Button("👥 Logout", elem_classes=["mobile-button"])

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
                new_state["semantic_memory_text"] = semantic_memory
                new_state["history"] = []
                new_state["user_memory_index"] = sessions_memory # یا هر نامی که ایندکس در آن است

                


                
                welcome_msg = hello_msg if hello_msg else f"Welcome back, {name}!"
                
                return (
                    gr.update(elem_classes=["center-content", "slide-up"]), # 2. افزودن کلاس انیمیشن به جای visible=False
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
                    "semantic_memory_text": semantic_memory,
                    "user_memory_index": sessions_memory,
                    "episodic_memory_index": episodic_memory,
                    "semantic_memory_index": semantic_memory,
                }
            )

            welcome_msg = f"Welcome {name}! Registration complete."

            return (
                gr.update(
                    elem_classes=["center-content", "slide-up"]
                ),
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
                gr.update(visible=True, elem_classes=["center-content"]), # 4. برداشتن کلاس انیمیشن تا در خروج، فرم لاگین دوباره دیده شود
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(value="Logged out successfully.", visible=True),
                gr.update(value=""),
                new_state,
                gr.update(value=[])
            )

        def handle_chat(user_message, state):
            if not user_message.strip():
                print("[handle_chat] empty message")
                print("[handle_chat] state =", state)
                return gr.update(), state.get("history", []), state

            print("[handle_chat] user_message =", repr(user_message))
            print("[handle_chat] state keys =", list(state.keys()) if isinstance(state, dict) else type(state))
            print("[handle_chat] user_name =", state.get("user_name"))
            print("[handle_chat] user_memory_index =", state.get("user_memory_index"))
            print("[handle_chat] memory exists =", state.get("memory") is not None)
            print("[handle_chat] service_context exists =", state.get("service_context") is not None)
            print("[handle_chat] api_index =", state.get("api_index", 0))
            print("[handle_chat] semantic_memory_text =", repr(state.get("semantic_memory_text", "")))
            print("[handle_chat] history len =", len(state.get("history", [])))

            user_name = state.get("user_name")
            user_memory_index = state.get("user_memory_index")
            user_memory = state.get("memory")
            service_context = state.get("service_context")
            api_index = state.get("api_index", 0)
            semantic_memory_text = state.get("semantic_memory_text", "")

            query_category = classify_query_local(user_message)
            print("[handle_chat] query_category =", query_category)

            new_history, _, status_msg = predict_new(
                text=user_message,
                history=state.get("history", []),
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
                query_category=query_category
            )

            print("[handle_chat] status_msg =", status_msg)
            print("[handle_chat] new_history len =", len(new_history))

            state["history"] = new_history
            print("[handle_chat] updated state history len =", len(state.get("history", [])))

            return "", new_history, state
            
        def clear_history(state):
            state["history"] = []
            return [], state
        
        def handle_new_session(state):
            # بررسی اینکه آیا کاربر لاگین کرده است یا خیر
            user_name = state.get("user_name")
            if not user_name or "memory" not in state or user_name not in state["memory"]:
                print("Error: User not logged in or memory missing.")
                # خروجی اول برای chatbot (خالی) و خروجی دوم برای state
                return [], state 

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
            
            # برگرداندن مقادیر با ترتیب درست برای خروجی‌های Gradio: [chatbot, state]
            return [], state

        check_user_btn.click(
            handle_login_check,
            inputs=[username_input, state],
            outputs=[login_page, registration_fields, chat_page, login_status, active_header, system_msg, state]
        )

        register_btn.click(
            handle_register,
            inputs=[username_input, age_input, gender_input, occupation_input, residence_input, state],
            outputs=[login_page, registration_fields, chat_page, login_status, active_header, system_msg, state]
        )

        switch_user_btn.click(
            switch_user,
            inputs=[state],
            outputs=[login_page, registration_fields, chat_page, login_status, username_input, state, chatbot]
        )
        
        submit_btn.click(
            handle_chat,
            inputs=[user_input, state],
            outputs=[user_input, chatbot, state]
        )
        
        user_input.submit(
            handle_chat,
            inputs=[user_input, state],
            outputs=[user_input, chatbot, state]
        )
        
        clear_btn.click(
            clear_history,
            inputs=[state],
            outputs=[chatbot, state]
        )
        
                # تغییر از clear_history به handle_new_session
        new_session_btn.click(
            handle_new_session,
            inputs=[state],
            outputs=[chatbot, state]
        

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
