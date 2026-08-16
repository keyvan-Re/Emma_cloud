import os
import json
import math
import copy
import random
import datetime
import numpy as np
from typing import List, Tuple, Optional

# LangChain Imports
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter

# Local Imports
# Note: Ensure these modules exist in your project structure
from memory_retrieval.textsplitterE import ChineseTextSplitter
from memory_retrieval.configs.model_config import *

# --- Constants ---
# Return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 6
# LLM input history length
LLM_HISTORY_LEN = 3


def forgetting_curve(t: float, S: float) -> float:
    """
    Calculate the retention of information at time t based on the Ebbinghaus forgetting curve.
    
    :param t: Time elapsed since the information was learned (in days).
    :param S: Strength of the memory. Higher S means slower forgetting.
    :return: Probability of retention (0.0 to 1.0).
    """
    # The formula models exponential decay of memory
    return math.exp(-t / (5 * S))


def get_docs_with_score(docs_with_score):
    """
    Helper function to unpack documents and attach similarity scores to metadata.
    """
    docs = []
    for doc, score in docs_with_score:
        doc.metadata["score"] = score
        docs.append(doc)
    return docs


def seperate_list(ls: List[int]) -> List[List[int]]:
    """
    Groups a list of integers into sub-lists of consecutive integers.
    Used to reconstruct continuous memory chunks.
    Example: [1, 2, 4, 5, 6] -> [[1, 2], [4, 5, 6]]
    """
    if not ls:
        return []
        
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i-1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists


class MemoryForgetterLoader(UnstructuredFileLoader):
    """
    A custom loader that handles loading memory logs, applying the forgetting curve,
    and saving the updated memory state back to the file.
    """
    def __init__(self, filepath, language, mode="elements"):
        super().__init__(filepath, mode=mode)
        self.filepath = filepath
        self.memory_level = 3
        self.total_date = 30
        self.language = language
        self.memory_bank = {}

    def _get_date_difference(self, date1: str, date2: str) -> int:
        """Calculates days difference between two date strings (YYYY-MM-DD)."""
        date_format = "%Y-%m-%d"
        d1 = datetime.datetime.strptime(date1, date_format)
        d2 = datetime.datetime.strptime(date2, date_format)
        return (d2 - d1).days

    def update_memory_when_searched(self, recalled_memos, user, cur_date):
        """
        Strengthens the memory when it is successfully recalled/searched.
        Updates 'memory_strength' and 'last_recall_date'.
        """
        for recalled in recalled_memos:
            recalled_id = recalled.metadata['memory_id']
            # Assuming ID format: user_date_index
            try:
                recalled_date = recalled_id.split('_')[1]
            except IndexError:
                continue

            if user in self.memory_bank and recalled_date in self.memory_bank[user]['history']:
                for i, memory in enumerate(self.memory_bank[user]['history'][recalled_date]):
                    if memory['memory_id'] == recalled_id:
                        # Increase strength and update date
                        self.memory_bank[user]['history'][recalled_date][i]['memory_strength'] += 1
                        self.memory_bank[user]['history'][recalled_date][i]['last_recall_date'] = cur_date
                        break
    
    def write_memories(self, out_file):
        """Saves the current state of memory bank to JSON file."""
        with open(out_file, "w", encoding="utf-8") as f:
            print(f'Successfully wrote to {out_file}')
            json.dump(self.memory_bank, f, ensure_ascii=False, indent=4)
    
    def load_memories(self, memory_file):
        with open(memory_file, "r", encoding="utf-8") as f:
            self.memory_bank = json.load(f)

    def initial_load_forget_and_save(self, name, now_date):
        """
        Loads memories, applies forgetting mechanism based on time, and prepares documents for vector store.
        """
        docs = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            memories = json.load(f)
            
            for user_name, user_memory in memories.items():
                # Filter by user if needed (currently commented out)
                # if user_name != name: continue

                if 'history' not in user_memory.keys():
                    continue
                
                self.memory_bank[user_name] = copy.deepcopy(user_memory)
                
                # Iterate through history by date
                for date, content in user_memory['history'].items():
                     # Define prefixes based on language
                     if self.language == 'cn':
                         memory_intro = f'Conversation content on time {date}:' # Translated from '时间{date}的对话内容：'
                         user_kw = '[|User|]:' # Translated
                         ai_kw = '[|AI Companion|]:' # Translated
                     else:
                         memory_intro = f'Conversation content on {date}:'
                         user_kw = '[|User|]:'
                         ai_kw = '[|AI|]:'

                     forget_ids = []
                     
                     for i, dialog in enumerate(content):
                        tmp_str = memory_intro
                        
                        # Normalize dialog format
                        if not isinstance(dialog, dict):
                            dialog = {'query': dialog[0], 'response': dialog[1]}
                            self.memory_bank[user_name]['history'][date][i] = dialog

                        query = dialog['query']
                        response = dialog['response']
                        
                        # Get memory metadata
                        memory_strength = dialog.get('memory_strength', 1)
                        last_recall_date = dialog.get('last_recall_date', date)
                        memory_id = dialog.get('memory_id', f'{user_name}_{date}_{i}')
                        
                        # Construct text for embedding
                        tmp_str += f'{user_kw} {query.strip()}; '
                        tmp_str += f'{ai_kw} {response.strip()}'
                        
                        metadata = {
                            'memory_strength': memory_strength,
                            'memory_id': memory_id,
                            'last_recall_date': last_recall_date,
                            "source": memory_id
                        }
                        
                        # Update memory bank with metadata
                        self.memory_bank[user_name]['history'][date][i].update(metadata)
                        
                        # Calculate retention
                        days_diff = self._get_date_difference(last_recall_date, now_date)
                        retention_probability = forgetting_curve(days_diff, memory_strength)
                        
                        # print(f"Diff: {days_diff}, Strength: {memory_strength}, Retention: {retention_probability}")

                        # Probabilistic forgetting
                        if random.random() > retention_probability:
                            forget_ids.append(i) # Mark for deletion
                        else:
                            docs.append(Document(page_content=tmp_str, metadata=metadata))
                     
                     # print(f"User: {user_name}, Date: {date}, Forgotten IDs: {forget_ids}")

                     # Remove forgotten memories
                     if len(forget_ids) > 0:
                         forget_ids.sort(reverse=True)
                         for idd in forget_ids:
                             self.memory_bank[user_name]['history'][date].pop(idd)
                     
                     # Clean up empty dates
                     if len(self.memory_bank[user_name]['history'][date]) == 0:
                            self.memory_bank[user_name]['history'].pop(date)
                            if date in self.memory_bank[user_name]['summary']:
                                self.memory_bank[user_name]['summary'].pop(date)
                                
                     # Handle Summaries
                     if 'summary' in self.memory_bank[user_name].keys():
                        if date in self.memory_bank[user_name]['summary'].keys():
                            summary_data = self.memory_bank[user_name]["summary"][date]
                            
                            if not isinstance(summary_data, dict):
                                self.memory_bank[user_name]["summary"][date] = {'content': summary_data}
                            
                            summary_str = self.memory_bank[user_name]["summary"][date]["content"] if isinstance(self.memory_bank[user_name]["summary"][date], dict) else self.memory_bank[user_name]["summary"][date]
                            
                            if self.language == 'cn':
                                summary_text = f'The summary of the conversation on time {date} is: {summary_str}' # Translated
                            else:
                                summary_text = f'The summary of the conversation on {date} is: {summary_str}'
                                
                            memory_strength = self.memory_bank[user_name]['summary'][date].get('memory_strength', 1) 
                            last_recall_date = self.memory_bank[user_name]["summary"][date].get('last_recall_date', date) 
                            
                            metadata = {
                                'memory_strength': memory_strength,
                                'memory_id': f'{user_name}_{date}_summary',
                                'last_recall_date': last_recall_date,
                                "source": f'{user_name}_{date}_summary'
                            }
                            
                            if isinstance(self.memory_bank[user_name]["summary"][date], dict):    
                                self.memory_bank[user_name]['summary'][date].update(metadata)
                            else:
                                self.memory_bank[user_name]['summary'][date] = {'content': self.memory_bank[user_name]['summary'][date], **metadata}
                            
                            docs.append(Document(page_content=summary_text, metadata=metadata))
                            
        # Save changes back to file
        self.write_memories(self.filepath) 
        return docs


def load_memory_file(filepath, user_name, cur_date='', language='cn'):
    """Wrapper to initialize loader and split documents."""
    memory_loader = MemoryForgetterLoader(filepath, language)
    
    # Use ChineseTextSplitter (or switch to standard if English)
    textsplitter = ChineseTextSplitter(pdf=False)
    
    formatted_path = filepath.replace('.json', '_forget_format.json')
    print('Writing to formatted path existence check:', os.path.exists(formatted_path))
    
    if not cur_date:
        cur_date = datetime.date.today().strftime("%Y-%m-%d")
     
    docs = memory_loader.initial_load_forget_and_save(user_name, cur_date)
    docs = textsplitter.split_documents(docs)
    
    return docs, memory_loader 
    

def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        """
        Custom search function to be attached to FAISS.
        It retrieves not just the matched vector, but also attempts to reconstruct 
        context by fetching surrounding chunks (neighbors) from the docstore.
        """
        scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        docs = []
        id_set = set()
        
        for j, i in enumerate(indices[0]):
            if i == -1:
                # This happens when not enough docs are returned.
                continue
            
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            id_set.add(i)
            docs_len = len(doc.page_content)
            
            # Expand context window (retrieve neighbor chunks)
            for k in range(1, max(i, len(self.index_to_docstore_id)-i)):
                for l in [i+k, i-k]:
                    if 0 <= l < len(self.index_to_docstore_id):
                        _id0 = self.index_to_docstore_id[l]
                        doc0 = self.docstore.search(_id0)
                        
                        if docs_len + len(doc0.page_content) > self.chunk_size:
                            break
                        
                        # Only merge if they come from the same source (same conversation/date)
                        elif doc0.metadata["source"] == doc.metadata["source"]:
                            docs_len += len(doc0.page_content)
                            id_set.add(l)
                            
        id_list = sorted(list(id_set))
        id_lists = seperate_list(id_list)
        
        for id_seq in id_lists:
            # Reconstruct the document from chunks
            for id in id_seq:
                if id == id_seq[0]:
                    _id = self.index_to_docstore_id[id]
                    doc = self.docstore.search(_id)
                else:
                    _id0 = self.index_to_docstore_id[id]
                    doc0 = self.docstore.search(_id0)
                    doc.page_content += doc0.page_content
            
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            
            # Use the score of the primary match
            docs.append((doc, scores[0][j]))
            
        return docs


class LocalMemoryRetrieval:
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE

    def init_cfg(self,
                 embedding_model: str = EMBEDDING_MODEL_CN,
                 embedding_device=EMBEDDING_DEVICE,
                 top_k=VECTOR_SEARCH_TOP_K,
                 language='cn'
                 ):
        self.language = language
        # Assuming embedding_model_dict is defined in configs
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[embedding_model],
                                                model_kwargs={'device': embedding_device})
        self.user = ''
        self.top_k = top_k
        self.memory_loader = None
        self.memory_path = ''
        
    def init_memory_vector_store(self,
                                    filepath: str or List[str],
                                    vs_path: str or os.PathLike = None,
                                    user_name: str = None,
                                    cur_date: str = None):
        """Initializes or loads the FAISS vector store with memory data."""
        loaded_files = []
        self.user = user_name
        
        docs = []

        # Handle Single File
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                print("Path does not exist")
                return None, None
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                self.memory_path = filepath
                new_docs, memory_loader = load_memory_file(filepath, user_name, cur_date, self.language)
                docs = new_docs
                print(f"{file} Loaded successfully")
                loaded_files.append(filepath)
            elif os.path.isdir(filepath):
                for file in os.listdir(filepath):
                    fullfilepath = os.path.join(filepath, file)
                    try:
                        self.memory_path = fullfilepath
                        new_docs, memory_loader = load_memory_file(fullfilepath, user_name, cur_date, self.language)
                        docs += new_docs
                        print(f"{file} Loaded successfully")
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        print(e)
                        print(f"{file} Failed to load")
        # Handle List of Files
        else:
            for file in filepath:
                try:
                    self.memory_path = file
                    new_docs, memory_loader = load_memory_file(file, user_name, cur_date, self.language)
                    docs += new_docs
                    print(f"{file} Loaded successfully")
                    loaded_files.append(file)
                except Exception as e:
                    print(e)
                    print(f"{file} Failed to load")

        self.memory_loader = memory_loader
        
        if len(docs) > 0:
            if vs_path and os.path.isdir(vs_path):
                # Load existing index and append
                vector_store = FAISS.load_local(vs_path, self.embeddings)
                print(f'Load from previous memory index {vs_path}.')
                vector_store.add_documents(docs)
            else:
                # Create new index
                if not vs_path:
                    filename = os.path.split(filepath)[-1] if isinstance(filepath, str) else "multi_files"
                    vs_path = f"""{VS_ROOT_PATH}{os.path.splitext(filename)[0]}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"""
                vector_store = FAISS.from_documents(docs, self.embeddings)
            
            print(f'Saving to {vs_path}')
            vector_store.save_local(vs_path)
            return vs_path, loaded_files
        else:
            print("No files loaded successfully. Please check dependencies or replace files.")
            return None, loaded_files
    
    def load_memory_index(self, vs_path):
        """Loads a pre-existing FAISS index and monkey-patches the search method."""
        vector_store = FAISS.load_local(vs_path, self.embeddings)
        # Monkey patch custom search method to FAISS instance
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_size = self.chunk_size
        return vector_store
    
    def search_memory(self, query, vector_store, cur_date=''):
        """Search for memories related to the query."""
        
        # Perform similarity search using the patched/custom method implied by usage
        related_docs_with_score = vector_store.similarity_search_with_score(query, k=self.top_k)
        
        related_docs = get_docs_with_score(related_docs_with_score)
        # Sort results by date/source order
        related_docs = sorted(related_docs, key=lambda x: x.metadata["source"], reverse=False)
        
        pre_date = ''
        date_docs = []
        dates = []
        cur_date = cur_date if cur_date else datetime.date.today().strftime("%Y-%m-%d") 
        
        for doc in related_docs:
            # Clean up content for display
            # Assuming format: "Conversation content on time {date}:" or "Conversation content on {date}:"
            clean_content = doc.page_content
            # Remove the header text if present (rudimentary cleanup based on known prefixes)
            clean_content = clean_content.split(':')[-1].strip() if ':' in clean_content else clean_content

            if doc.metadata["source"] != pre_date:
                date_docs.append(clean_content)
                pre_date = doc.metadata["source"]
                dates.append(pre_date)
            else:
                date_docs[-1] += f'\n{clean_content}' 
        
        # Update memory strength since they were recalled
        self.memory_loader.update_memory_when_searched(related_docs, user=self.user, cur_date=cur_date)
        self.save_updated_memory()
        
        return date_docs, ', '.join(dates) 
    
    def save_updated_memory(self):
        self.memory_loader.write_memories(self.memory_path)
