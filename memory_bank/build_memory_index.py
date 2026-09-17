# -*- coding:utf-8 -*-
#
# Memory Indexing Utility for LlamaIndex
# Project: EMMA (Empathetic Memory-augmented Multi-layer Assistant)
#

# --- Standard Library Imports ---
import json
import os

# --- Third-Party Imports ---
import tiktoken
from llama_index.core import (
    Document,
    PromptHelper,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Global Variables & Paths ---
index_set = {}

BASE_DATA_DIR = os.getenv(
    "EMMA_DATA_DIR", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
)
_INDEX_BASE_DIR = os.getenv(
    "EMMA_INDEX_DIR", 
    os.path.join(BASE_DATA_DIR, "memories", "memory_index", "llamaindex")
)
GAPGPT_BASE_URL = os.getenv("GAPGPT_BASE_URL", "https://api.gapgpt.app/v1")


# --- Core Configuration Functions ---

def setup_global_settings():
    """
    Configures global settings for LlamaIndex (v0.10+).
    Sets LLM and Embedding models to route through GapGPT.
    """
    GAPGPTMASKTOKENwdfn4ezolb9X0X = os.getenv("GAPGPT_API_KEY") or os.getenv("OPENAI_API_KEY")

    # 1. Set Tokenizer
    try:
        Settings.tokenizer = tiktoken.get_encoding("cl100k_base").encode
    except Exception as e:
        print(f"Warning: Could not set tokenizer: {e}")

    # 2. Configure LLM for GapGPT
    Settings.llm = OpenAI(
        model="gpt-4o",
        temperature=1.0,
        max_tokens=1024,
        api_key=GAPGPTMASKTOKENwdfn4ezolb9X1X,
        api_base=GAPGPT_BASE_URL,
        additional_kwargs={
            "top_p": 0.95,
            "frequency_penalty": 0.4,
            "presence_penalty": 0.2
        }
    )

    # 3. Configure Embedding Model for GapGPT
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002",
        api_key=GAPGPTMASKTOKENwdfn4ezolb9X2X,
        api_base=GAPGPT_BASE_URL
    )

    # 4. Configure Prompt Helper
    Settings.prompt_helper = PromptHelper(
        context_window=4096,
        num_output=256,
        chunk_overlap_ratio=0.1
    )

    # 5. Chunk Settings
    Settings.chunk_size = 512
    Settings.chunk_overlap = 20


# --- Document Generation Functions ---

def generate_memory_docs(data, language="en"):
    """
    Generates structured LlamaIndex Document objects from nested user memory dictionary.
    """
    all_user_memories = {}

    for user_name, user_memory in data.items():
        all_user_memories[user_name] = {
            "sessions": [],
            "episodic_memory": [],
            "semantic_memory": [],
        }

        # 1. Process session history
        if "sessions" in user_memory and user_memory["sessions"]:
            for session in user_memory["sessions"]:
                date = session.get("date", "")
                content = session.get("conversation", [])

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

                all_user_memories[user_name]["sessions"].append(Document(text=memory_str))

        # 2. Process episodic memory
        if "episodic_memory" in user_memory and user_memory["episodic_memory"]:
            episodic_str = "Recent experiences:\n"
            for event in user_memory["episodic_memory"]:
                episodic_str += f"- {event}\n"
            all_user_memories[user_name]["episodic_memory"].append(Document(text=episodic_str))

        # 3. Process semantic memory
        if "semantic_memory" in user_memory and user_memory["semantic_memory"]:
            semantic_str = "Long-term personality traits and facts:\n"
            if isinstance(user_memory["semantic_memory"], dict):
                for trait, value in user_memory["semantic_memory"].items():
                    semantic_str += f"- {trait}: {value}\n"
            elif isinstance(user_memory["semantic_memory"], list):
                for item in user_memory["semantic_memory"]:
                    semantic_str += f"- {item}\n"
            else:
                print(f"WARNING: semantic_memory for '{user_name}' has unexpected format: {type(user_memory['semantic_memory'])}")

            all_user_memories[user_name]["semantic_memory"].append(Document(text=semantic_str))

    return all_user_memories


# --- Indexing & Storage Functions ---

def build_memory_index(all_user_memories, data_args=None, name=None):
    """
    Build and persist memory indices for each user and memory layer.
    """
    language = getattr(data_args, 'language', 'en') if data_args else 'en'
    structured_docs = generate_memory_docs(all_user_memories, language=language)

    setup_global_settings()

    for user_name, memories_by_type in structured_docs.items():
        if name and user_name != name:
            continue

        print(f"Building indices for user '{user_name}'...")

        base_path = os.path.join(_INDEX_BASE_DIR, user_name)

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


def load_memory_index(user_name, memory_type):
    """
    Loads a specific persisted VectorStoreIndex from disk.
    """
    setup_global_settings()
    
    key = f"{user_name}_{memory_type}"
    if key in index_set:
        return index_set[key]

    index_dir = os.path.join(_INDEX_BASE_DIR, user_name, memory_type)
    if not os.path.exists(index_dir):
        return None

    try:
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        index = load_index_from_storage(storage_context)
        index_set[key] = index
        return index
    except Exception as e:
        print(f"Error loading index for {key} from {index_dir}: {e}")
        return None


def load_all_user_indices(user_name):
    """
    Loads all available memory layers (sessions, episodic, semantic) for a given user.
    """
    memory_types = ["sessions", "episodic_memory", "semantic_memory"]
    loaded = {}
    for m_type in memory_types:
        idx = load_memory_index(user_name, m_type)
        if idx:
            loaded[m_type] = idx
    return loaded
