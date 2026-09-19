#
# Memory Indexing Utility for LlamaIndex
#
# This script provides functionalities to build and manage memory indices for a conversational AI
# using the LlamaIndex framework. It supports creating structured memory documents from raw
# conversation data and persisting them as vector indices for efficient retrieval.
#

# --- Standard Library Imports ---
import json
import os

# --- Third-Party Imports ---
import tiktoken
from llama_index.core import (
    Document,
    PromptHelper,
    Settings,           # Replaces the deprecated ServiceContext
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
# If a specific embedding model is needed (e.g., from OpenAI), uncomment the line below
# from llama_index.embeddings.openai import OpenAIEmbedding
# --- PATH CONFIGURATION (CRITICAL UPDATE) ---
# پیدا کردن مسیر پایه بر اساس موقعیت فایل اسکریپت برای اطمینان از ذخیره‌سازی صحیح
# فرض بر این است که فایل در /app/utils/ یا /app/memory_bank/ است
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, ".."))

# تنظیم مسیر دقیق برای ذخیره ایندکس‌ها
# این مسیر باید دقیقاً همان مسیری باشد که memory_utils.py آن را آپلود می‌کند
MEMORIES_DIR = os.path.join(BASE_DIR, "memories")
INDEX_BASE_DIR = os.path.join(MEMORIES_DIR, "memory_index")
_LLAMAINDEX_BASE_DIR = os.path.join(INDEX_BASE_DIR, "llamaindex")


print(f"📊 Index Builder Path Config:")
print(f"   - Script Dir: {CURRENT_SCRIPT_DIR}")
print(f"   - Index Target Dir: {INDEX_BASE_DIR}")

# --- Global Variables ---
# A dictionary to hold the loaded or newly created indices in memory.
index_set = {}





# --- Core Functions ---

def setup_global_settings():
    """
    Configures global settings for LlamaIndex (v0.10+).

    This function replaces the deprecated `build_service_context` pattern by directly
    configuring the global `Settings` object for the LLM, tokenizer, and other components.
    """
    # 1. Set Tokenizer (to resolve potential tiktoken errors)
    try:
        # This encoding is commonly used by OpenAI models like GPT-4.
        Settings.tokenizer = tiktoken.get_encoding("cl100k_base").encode
    except Exception as e:
        print(f"Warning: Could not set the tokenizer. This might lead to issues. Error: {e}")

    # 2. Configure the Language Model (LLM)
    # Use the LlamaIndex-specific OpenAI class to avoid conflicts with other libraries.
    # It automatically reads the API key from the "OPENAI_API_KEY" environment variable.
    Settings.llm = OpenAI(
        model="gpt-4o",
        temperature=1.0,
        max_tokens=1024,
        api_key=os.environ.get("GAPGPT_API_KEY"),
        api_base="https://api.gapgpt.app/v1",
        additional_kwargs={
            "top_p": 0.95,
            "frequency_penalty": 0.4,
            "presence_penalty": 0.2
        }
    )

    # 3. Configure the Prompt Helper (with updated parameter names)
    # This helps manage context window size, number of outputs, etc.
    Settings.prompt_helper = PromptHelper(
        context_window=4096,
        num_output=256,
        chunk_overlap_ratio=0.1
    )

    # 4. Configure Chunk Size for document processing
    Settings.chunk_size = 512
    Settings.chunk_overlap = 20

    # Optional: Configure the embedding model if needed.
    # If not set, LlamaIndex defaults to a compatible model (e.g., OpenAI's text-embedding-ada-002).
    # Settings.embed_model = OpenAIEmbedding(api_key=GAPGPTMASKTOKENlgicihcumgiX0X, api_base=GAPGPTMASKTOKENlgicihcumgiX1X)


def generate_memory_docs(data, language):
    """
    Generates structured LlamaIndex Document objects from a nested dictionary of user memories.

    This function processes different types of memory (sessions, episodic, semantic) and
    formats them into Document objects suitable for indexing.

    Args:
        data (dict): A dictionary where keys are user names and values contain their memories.
        language (str): The language code (e.g., "en"), currently used for formatting.

    Returns:
        dict: A dictionary where keys are user names and values are dicts containing
              lists of Document objects for each memory type.
    """
    all_user_memories = {}

    for user_name, user_memory in data.items():
        all_user_memories[user_name] = {
            "sessions": [],
            "episodic_memory": [],
            "semantic_memory": [],
        }

        # Process session history
        if "sessions" in user_memory:
            for session in user_memory["sessions"]:
                date = session["date"]
                content = session["conversation"]

                memory_str = f"Session on {date}:\n"
                for dialog in content:
                    query = dialog.get("query", "")
                    response = dialog.get("response", "")
                    memory_str += f"\n{user_name}: {query.strip()}"
                    memory_str += f"\nAI: {response.strip()}"
                memory_str += "\n"

                if "summary" in user_memory and date in user_memory["summary"]:
                    summary = f'The summary of the conversation on {date} is: {user_memory["summary"][date]}'
                    memory_str += summary

                # Use the 'text' argument for the Document object
                all_user_memories[user_name]["sessions"].append(Document(text=memory_str))

        # Process episodic memory
        if "episodic_memory" in user_memory:
            episodic_str = "Recent experiences:\n"
            for event in user_memory["episodic_memory"]:
                episodic_str += f"- {event}\n"
            all_user_memories[user_name]["episodic_memory"].append(Document(text=episodic_str))

        # Process semantic memory (long-term facts and personality)
        if "semantic_memory" in user_memory:
            semantic_str = "Long-term personality traits and facts:\n"
            if isinstance(user_memory["semantic_memory"], dict):
                for trait, value in user_memory["semantic_memory"].items():
                    semantic_str += f"- {trait}: {value}\n"
            elif isinstance(user_memory["semantic_memory"], list):
                for item in user_memory["semantic_memory"]:
                    semantic_str += f"- {item}\n"
            else:
                print(f"WARNING: semantic_memory for user '{user_name}' has an unexpected format: {type(user_memory['semantic_memory'])}")

            all_user_memories[user_name]["semantic_memory"].append(Document(text=semantic_str))

    return all_user_memories


def build_memory_index(all_user_memories, data_args, name=None):
    """
    Build and persist memory indices for each user and memory type.
    """
    structured_docs = generate_memory_docs(
        all_user_memories, data_args.language
    )

    setup_global_settings()

    for user_name, memories_by_type in structured_docs.items():
        if name and user_name != name:
            continue

        print(f"Building indices for user '{user_name}'")

        base_path = os.path.join(_LLAMAINDEX_BASE_DIR, user_name)

        path_map = {
            "sessions": os.path.join(base_path, "sessions"),
            "episodic_memory": os.path.join(base_path, "episodic_memory"),
            "semantic_memory": os.path.join(base_path, "semantic_memory"),
        }

        for memory_type, docs in memories_by_type.items():
            if not docs:
                print(f"  -> Skipping '{memory_type}' index (no documents found).")
                continue

            if memory_type not in path_map:
                print(f"  -> Skipping unknown memory type: '{memory_type}'")
                continue

            print(f"  -> Building '{memory_type}' index...")

            cur_index = VectorStoreIndex.from_documents(docs)

            index_dir = path_map[memory_type]
            os.makedirs(index_dir, exist_ok=True)

            cur_index.storage_context.persist(persist_dir=index_dir)
            print(f"  + Saved '{memory_type}' index to: {index_dir}")

            index_set[f"{user_name}_{memory_type}"] = cur_index


# --- Deprecated Functions (Kept for reference) ---

def generate_memory_docs_old(data, language):
    """
    DEPRECATED: An older version of the document generation function.

    This function had a simpler structure and contained Chinese string literals.
    It is replaced by `generate_memory_docs`.
    """
    all_user_memories = {}
    for user_name, user_memory in data.items():
        all_user_memories[user_name] = []
        if "history" not in user_memory:
            continue
        for date, content in user_memory["history"].items():
            memory_str = f"Conversation on {date}:"

            for dialog in content:
                query = dialog.get("query", "")
                response = dialog.get("response", "")
                memory_str += f"\n{user_name}: {query.strip()}"
                memory_str += f"\nAI: {response.strip()}"
            memory_str += "\n"
            if "summary" in user_memory and date in user_memory["summary"]:
                summary = f'The summary of the conversation on {date} is: {user_memory["summary"][date]}'
                memory_str += summary

            all_user_memories[user_name].append(Document(text=memory_str))
    return all_user_memories


def build_memory_index_old(all_user_memories, data_args, name=None):
    """
    DEPRECATED: An older version of the index building function.

    This function used a flatter memory structure and is replaced by the more
    comprehensive `build_memory_index`.
    """
    # The function name is a typo in the original code, corrected here for clarity.
    all_user_memories_docs = generate_memory_docs(
        all_user_memories, data_args.language
    )

    # Apply global settings
    setup_global_settings()

    for user_name, memories in all_user_memories_docs.items():
        if name and user_name != name:
            continue
        print(f"Building index for user {user_name} (using old method)")

        # The service_context argument is removed in newer versions.
        cur_index = VectorStoreIndex.from_documents(memories)
        index_set[user_name] = cur_index

                # استفاده از مسیر داینامیک و امن
        save_dir = os.path.join(_LLAMAINDEX_BASE_DIR, f"{user_name}_store")
        os.makedirs(save_dir, exist_ok=True)
        cur_index.storage_context.persist(persist_dir=save_dir)

