import os
import sys
import json
import time
import datetime
import shutil
import gradio as gr
from pprint import pprint
from huggingface_hub import HfApi

import traceback

# LlamaIndex Imports
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex



import traceback

try:
    from memory_bank.build_memory_index import build_memory_index

    from memory_bank.summarize_memory import (
        summarize_memory,
        extract_session_summary,
        extract_semantic_memory,
    )

except ImportError:
    print(
        "CRITICAL: Failed to import memory_bank modules.",
        flush=True,
    )
    traceback.print_exc()
    raise



# ==========================================
# 1. تنظیمات مسیر و هاب (Cloud Config)
# ==========================================

REPO_ID = "Keyvan1986/Emma-memory-storage"
REPO_TYPE = "dataset"
HF_TOKEN = os.environ.get("HF_TOKEN")

# مسیرهای اصلی (Absolute Paths)
BASE_DIR = os.path.abspath(os.getcwd())  # معمولاً /app
MEMORIES_DIR = os.path.join(BASE_DIR, "memories")
MEMORY_INDEX_DIR_NAME = "memory_index"
MEMORY_INDEX_PATH = os.path.join(MEMORIES_DIR, MEMORY_INDEX_DIR_NAME)
_LLAMAINDEX_BASE_DIR = os.path.join(MEMORY_INDEX_PATH, "llamaindex")

MEMORY_FILE_NAME = "update_memory_0512_eng.json"
MEMORY_FILE_PATH = os.path.join(MEMORIES_DIR, MEMORY_FILE_NAME)

# --- مسیر "اشتباه" قدیمی که فایل‌ها آنجا ساخته می‌شوند ---
# لاگ شما نشان داد فایل‌ها اینجا می‌روند: ../memories
LEGACY_OUTSIDE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "memories"))
LEGACY_INDEX_PATH = os.path.join(LEGACY_OUTSIDE_DIR, MEMORY_INDEX_DIR_NAME)

os.makedirs(MEMORIES_DIR, exist_ok=True)
os.makedirs(MEMORY_INDEX_PATH, exist_ok=True)

print(f"📂 Path Configuration:")
print(f"   - App Dir (Target): {MEMORY_INDEX_PATH}")
print(f"   - Builder Dir (Source): {LEGACY_INDEX_PATH}")

# ------------------------------------------------------------------------


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
        memory_index_path = os.path.join(_LLAMAINDEX_BASE_DIR, name)
        os.makedirs(memory_index_path, exist_ok=True)
        
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
def upload_user_indices_to_hf(name):
    local_dir = os.path.join(_LLAMAINDEX_BASE_DIR, name)
    remote_dir = f"memory_index/llamaindex/{name}"

    if not os.path.isdir(local_dir):
        raise FileNotFoundError(
            f"Index directory does not exist: {local_dir}"
        )

    files_count = sum(
        len(files)
        for _, _, files in os.walk(local_dir)
    )

    if files_count == 0:
        raise RuntimeError(
            f"No index files found for upload: {local_dir}"
        )

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "Hugging Face token is missing; index upload cannot run."
        )

    print(
        f"[HF] Uploading {files_count} index files\n"
        f"[HF] Local: {local_dir}\n"
        f"[HF] Dataset: {REPO_ID}\n"
        f"[HF] Remote: {remote_dir}",
        flush=True,
    )

    api = HfApi(token=token)

    try:
        commit_info = api.upload_folder(
            repo_id=REPO_ID,
            repo_type="dataset",
            folder_path=local_dir,
            path_in_repo=remote_dir,
            commit_message=f"Upload memory indices for {name}",
        )
    except Exception:
        print(
            f"[HF] Index upload FAILED for {name}",
            flush=True,
        )
        raise

    print(
        f"[HF] Index upload completed for {name}. "
        f"Commit: {commit_info.oid}",
        flush=True,
    )

    return commit_info

def enter_name_llamaindex(
    name,
    memory,
    data_args,
    update_memory_index=True
):
    """
    Load the user's session, episodic, and semantic memory indices.
    Compatible with LlamaIndex v0.10+.
    """

    sessions_memory = None
    episodic_memory = None
    semantic_memory = None

    print(f"[DEBUG] enter_name_llamaindex called with name={name!r}")

    if name not in memory:
        print(f"[DEBUG] user {name!r} not found in memory")
        return "User not found.", None, None, None, None

    user_memory = memory[name]

    # مسیر دقیق کاربر در پوشه llamaindex
    base_path = os.path.join(_LLAMAINDEX_BASE_DIR, name)

    sessions_path = os.path.join(
        base_path,
        "sessions"
    )

    episodic_path = os.path.join(
        base_path,
        "episodic_memory"
    )

    semantic_path = os.path.join(
        base_path,
        "semantic_memory"
    )

    print(f"[DEBUG] base_path: {base_path}")
    print(f"[DEBUG] sessions_path: {sessions_path}")
    print(f"[DEBUG] episodic_path: {episodic_path}")
    print(f"[DEBUG] semantic_path: {semantic_path}")

    indices_exist = (
        os.path.isdir(sessions_path)
        and os.path.isdir(episodic_path)
        and os.path.isdir(semantic_path)
    )

    print(f"[DEBUG] indices_exist: {indices_exist}")

    # ============================================================
    # Build / Update indices
    # ============================================================

    if update_memory_index or not indices_exist:

        print(
            f"[DEBUG] Initializing memory indices "
            f"for {name!r}..."
        )

        # 1. Build and persist LlamaIndex indices
        build_memory_index(
            memory,
            data_args,
            name=name
        )

        # ========================================================
        # 2. Upload persisted indices to Hugging Face Dataset
        # ========================================================

        try:
            print(
                f"[DEBUG] Uploading persisted indices "
                f"for user {name!r} to Hugging Face..."
            )

            upload_user_indices_to_hf(name)

            print(
                f"[DEBUG] Successfully uploaded indices "
                f"for {name!r}."
            )

        except Exception as exc:

            print(
                f"[ERROR] Failed to upload indices "
                f"for {name!r}: {exc}"
            )

            traceback.print_exc()

        # ========================================================
        # 3. Verify local persisted directories
        # ========================================================

        print("[DEBUG] After build:")

        print(
            f"[DEBUG] sessions_path exists: "
            f"{os.path.isdir(sessions_path)}"
        )

        print(
            f"[DEBUG] episodic_path exists: "
            f"{os.path.isdir(episodic_path)}"
        )

        print(
            f"[DEBUG] semantic_path exists: "
            f"{os.path.isdir(semantic_path)}"
        )

    # ============================================================
    # Safe index loading
    # ============================================================

    def load_index_safe(index_name, index_path):

        if not os.path.isdir(index_path):

            print(
                f"[DEBUG] {index_name} path does not exist: "
                f"{index_path}"
            )

            return None

        try:

            storage_context = StorageContext.from_defaults(
                persist_dir=index_path
            )

            index = load_index_from_storage(
                storage_context
            )

            print(
                f"[DEBUG] {index_name} loaded: "
                f"{type(index).__name__}, "
                f"is_none={index is None}"
            )

            return index

        except Exception as exc:

            print(
                f"[ERROR] Could not load "
                f"{index_name}: {exc}"
            )

            traceback.print_exc()

            return None

    # ============================================================
    # Load all three indices
    # ============================================================

    sessions_memory = load_index_safe(
        "sessions_memory",
        sessions_path
    )

    episodic_memory = load_index_safe(
        "episodic_memory",
        episodic_path
    )

    semantic_memory = load_index_safe(
        "semantic_memory",
        semantic_path
    )

    # ============================================================
    # Debug information
    # ============================================================

    print("[DEBUG] RETURN VALUES:")

    print(
        f"  hello_msg: Welcome back, {name}!"
    )

    print(
        f"  user_memory type: "
        f"{type(user_memory).__name__}"
    )

    print(
        f"  sessions_memory type: "
        f"{type(sessions_memory).__name__}"
    )

    print(
        f"  episodic_memory type: "
        f"{type(episodic_memory).__name__}"
    )

    print(
        f"  semantic_memory type: "
        f"{type(semantic_memory).__name__}"
    )

    return (
        f"Welcome back, {name}!",
        user_memory,
        sessions_memory,
        episodic_memory,
        semantic_memory,
    )
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
        memory[user_name]["semantic_memory"] = extract_semantic_memory(
            memory[user_name]["semantic_memory"], 
            current_session["conversation"]
     )
    
    semantic_memory_text = memory[user_name]["semantic_memory"]
    
    # Save to file
    # Ensure memory directory exists
    os.makedirs(os.path.dirname(memory_dir), exist_ok=True)
    
    with open(memory_dir, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=4)

    return memory, semantic_memory_text