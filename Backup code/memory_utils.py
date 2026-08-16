import os
import sys
import json
import time
import datetime
import shutil
import gradio as gr
from pprint import pprint

# LlamaIndex Imports
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex

# Local Imports setup
# Assuming this file is in 'utils/', we step back to find 'memory_bank'
bank_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../memory_bank')
sys.path.append(bank_path)

# Import functions from sibling/child modules
# Ensure these files exist in the appended path
try:
    from build_memory_index import build_memory_index
    from summarize_memory import summarize_memory, extract_session_summary, extract_semantic_memory
except ImportError:
    # Fallback or placeholder if running independently for testing
    print("Warning: Could not import memory build/summary modules.")
    def build_memory_index(*args, **kwargs): pass
    def summarize_memory(*args, **kwargs): return {}
    def extract_session_summary(*args, **kwargs): return {}
    def extract_semantic_memory(*args, **kwargs): return {}

# --- Global Config for Paths ---
# Keeping the specific path requested by user
MEMORY_FILE_PATH = os.path.expanduser("C:\\Users\\keyva\\MMPL_gpt\\memories\\update_memory_0512_eng.json")


def enter_name(name, memory, local_memory_qa, data_args, update_memory_index=True):
    """
    Legacy/Basic function to load user memory and initialize vector store.
    """
    cur_date = datetime.date.today().strftime("%Y-%m-%d")
    user_memory_index = None
    
    # Handle Gradio States
    if isinstance(data_args, gr.State): data_args = data_args.value
    if isinstance(memory, gr.State): memory = memory.value
    if isinstance(local_memory_qa, gr.State): local_memory_qa = local_memory_qa.value
    
    memory_dir = MEMORY_FILE_PATH
    
    if name in memory.keys():
        user_memory = memory[name]
        memory_index_path = os.path.join(data_args.memory_basic_dir, f'memory_index/{name}_index')
        os.makedirs(os.path.dirname(memory_index_path), exist_ok=True)
        
        if (not os.path.exists(memory_index_path)) or update_memory_index:
            print(f'Initializing memory index {memory_index_path}...')
        
            if os.path.exists(memory_index_path):
                shutil.rmtree(memory_index_path)
            # Initialize using the local QA object
            memory_index_path, _ = local_memory_qa.init_memory_vector_store(
                filepath=memory_dir, 
                vs_path=memory_index_path, 
                user_name=name, 
                cur_date=cur_date
            )                      
        
        user_memory_index = local_memory_qa.load_memory_index(memory_index_path) if memory_index_path else None
        msg = f"Welcome back, {name}!"
        return msg, user_memory, memory, name, user_memory_index
    else:
        memory[name] = {}
        memory[name].update({"name": name}) 
        msg = f"Welcome, new user {name}! I will remember your name, so next time we meet, I'll be able to call you by your name!"
        return msg, memory[name], memory, name, user_memory_index


def enter_name_llamaindex(name, memory, data_args, update_memory_index=True):
    """
    Loads user memory and separate indices for each memory type (sessions, episodic, semantic).
    Compatible with LlamaIndex v0.10+
    """
    sessions_memory = None
    episodic_memory = None
    semantic_memory = None

    if name in memory.keys():
        user_memory = memory[name]

        # Set the base path
        base_path = os.path.join(data_args.memory_basic_dir, f'../memories/memory_index/llamaindex/{name}')
        
        # Define paths (باید به پوشه‌ها اشاره کنند، نه فایل)
        sessions_path = f"{base_path}/sessions" # یا نام دقیق پوشه شما
        episodic_path = f"{base_path}/episodic_memory"
        semantic_path = f"{base_path}/semantic_memory"

        # Check if directories exist to decide on rebuilding
        indices_exist = (
            os.path.isdir(sessions_path) and 
            os.path.isdir(episodic_path) and 
            os.path.isdir(semantic_path)
        )

        # If indices do not exist or need updating, rebuild them
        if update_memory_index or not indices_exist:
            print(f'Initializing memory indices for {name}...')
            build_memory_index(memory, data_args, name=name)

        # --- Modified Section: Loading with new LlamaIndex method ---
        
        # 1. Load Sessions Memory
        if os.path.exists(sessions_path):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=sessions_path)
                sessions_memory = load_index_from_storage(storage_context)
                print(f'  → Loaded sessions memory index for {name}.')
            except Exception as e:
                print(f"Warning: Could not load sessions memory: {e}")
        
        # 2. Load Episodic Memory
        if os.path.exists(episodic_path):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=episodic_path)
                episodic_memory = load_index_from_storage(storage_context)
                print(f'  → Loaded episodic memory index for {name}.')
            except Exception as e:
                print(f"Warning: Could not load episodic memory: {e}")
        
        # 3. Load Semantic Memory
        if os.path.exists(semantic_path):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=semantic_path)
                semantic_memory = load_index_from_storage(storage_context)
                print(f'  → Loaded semantic memory index for {name}.')
            except Exception as e:
                print(f"Warning: Could not load semantic memory: {e}")

        return f"Welcome back, {name}!", user_memory, sessions_memory, episodic_memory, semantic_memory

    else:
        # New User Initialization
        memory[name] = {"name": name} 
        return f"Welcome, new user {name}! I'll remember your name for next time.", memory[name], None, None, None


def summarize_memory_event_personality(data_args, memory, user_name):
    """
    Summarizes the memory and returns the user-specific memory dict.
    """
    if isinstance(data_args, gr.State): data_args = data_args.value
    if isinstance(memory, gr.State): memory = memory.value
    
    memory_dir = MEMORY_FILE_PATH
    
    # Note: Ensure summarize_memory handles the language argument correctly (passed 'en' or similar)
    memory = summarize_memory(memory_dir, user_name, language=data_args.language)
    user_memory = memory[user_name] if user_name in memory.keys() else {}
    return user_memory


def save_local_memory(memory, history, user_name, data_args, new_conversation=False):
    """
    Saves user-model conversations into memory and adds episodic memory for each session.
    Handles both list-of-lists (old Gradio) and list-of-dicts (new Gradio/OpenAI) formats.
    """
    if isinstance(data_args, gr.State): data_args = data_args.value
    if isinstance(memory, gr.State): memory = memory.value

    memory_dir = MEMORY_FILE_PATH

    # 1. Initialize user memory with ALL required structures FIRST
    if user_name not in memory:
        memory[user_name] = {
            "sessions": [],
            "episodic_memory": [],
            "semantic_memory": {}
        }
    
    # 2. Ensure all sub-structures exist and are correct type
    memory[user_name].setdefault("sessions", [])
    memory[user_name].setdefault("episodic_memory", [])
    memory[user_name].setdefault("semantic_memory", {})

    # 3. Now safely check types
    if not isinstance(memory[user_name]["semantic_memory"], dict):
        memory[user_name]["semantic_memory"] = {}

    # Create new session or update existing one
    if new_conversation or not memory[user_name]["sessions"]:
        if new_conversation and memory[user_name]["sessions"]:
            # Logic to summarize previous session before starting new one could go here
            pass 

        # Create new session
        session = {
            "session_id": len(memory[user_name]["sessions"]),
            "date": time.strftime("%Y-%m-%d", time.localtime()),
            "conversation": []
        }
        memory[user_name]["sessions"].append(session)

    current_session = memory[user_name]["sessions"][-1]
    
    # --- Modified section to fix KeyError: 0 and handle History formats ---
    if not new_conversation and history:
        last_item = history[-1]
        
        # Case 1: New Format (List of Dicts)
        # In this format, history is linear. The last item is bot response, second to last is user query.
        if isinstance(last_item, dict):
            if len(history) >= 2:
                user_query = history[-2].get('content', '')
                bot_response = history[-1].get('content', '')
                
                # Verify roles to ensure correct pairing
                if history[-2].get('role') == 'user' and history[-1].get('role') == 'assistant':
                    current_session["conversation"].append({
                        'query': user_query, 
                        'response': bot_response
                    })
                
        # Case 2: Old Format (List of Lists/Tuples)
        elif isinstance(last_item, (list, tuple)):
            current_session["conversation"].append({
                'query': last_item[0], 
                'response': last_item[1]
            })
    # ----------------------------------------------------------------------

    # Optional: Update semantic memory in real-time
    # memory[user_name]["semantic_memory"] = extract_semantic_memory(
    #    memory[user_name]["semantic_memory"], 
    #    current_session["conversation"]
    # )
    
    semantic_memory_text = memory[user_name]["semantic_memory"]
    
    # Save to file
    # Ensure memory directory exists
    os.makedirs(os.path.dirname(memory_dir), exist_ok=True)
    
    with open(memory_dir, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=4)

    return memory, semantic_memory_text
