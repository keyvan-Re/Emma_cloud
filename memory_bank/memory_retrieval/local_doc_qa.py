## Refactored Code

import datetime
import json
import os
from typing import List, Optional, Tuple

import numpy as np
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import (
    NLTKTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)
from langchain.vectorstores import FAISS

from memory_retrieval.configs.model_config import *
from memory_retrieval.textsplitterE import ChineseTextSplitter

VECTOR_SEARCH_TOP_K = 3


class JsonMemoryLoader(UnstructuredFileLoader):
    def __init__(self, filepath, language, mode="elements"):
        super().__init__(filepath, mode=mode)
        self.filepath = filepath
        self.language = language

    def _get_metadata(self, date: str) -> dict:
        return {"source": date}

    def load(self, name):
        user_memories = []
        print(self.file_path)
        with open(self.filepath, "r", encoding="utf-8") as f:
            memories = json.loads(f.read())
        for user_name, user_memory in memories.items():
            if user_name != name:
                continue
            user_memories = []
            if "history" not in user_memory.keys():
                continue
            for date, content in user_memory["history"].items():
                metadata = self._get_metadata(date)
                memory_str = (
                    f"Conversation content on {date}:"
                    if self.language != "cn"
                    else f"Conversation content on {date}:"
                )
                user_kw = "[|User|]:" if self.language != "cn" else "[|User|]:"
                ai_kw = "[|AI|]:" if self.language != "cn" else "[|AI|]:"
                for i, dialog in enumerate(content):
                    query, response = dialog["query"], dialog["response"]
                    tmp_str = memory_str
                    tmp_str += f"{user_kw} {query.strip()}; "
                    tmp_str += f"{ai_kw} {response.strip()}"
                    user_memories.append(Document(page_content=tmp_str, metadata=metadata))
                if "summary" in user_memory.keys():
                    if date in user_memory["summary"].keys():
                        summary = (
                            f"The summary of the conversation on {date} is: {user_memory['summary'][date]}"
                        )
                        user_memories.append(Document(page_content=summary, metadata=metadata))
        return user_memories

    def load_and_split(
        self, text_splitter: Optional[TextSplitter] = None, name=""
    ) -> List[Document]:
        """Load documents and split into chunks."""
        if text_splitter is None:
            _text_splitter: TextSplitter = RecursiveCharacterTextSplitter()
        else:
            _text_splitter = text_splitter
        docs = self.load(name)
        results = _text_splitter.split_documents(docs)
        return results


def load_file(filepath, language="cn"):
    if filepath.endswith(".md"):
        loader = UnstructuredFileLoader(filepath, mode="elements")
        docs = loader.load()
    elif filepath.endswith(".pdf"):
        loader = UnstructuredFileLoader(filepath)
        if language == "cn":
            textsplitter = ChineseTextSplitter(pdf=True)
        else:
            textsplitter = RecursiveCharacterTextSplitter(
                pdf=True, separators=["\n\n", "\n", " ", "", "Round"]
            )
        docs = loader.load_and_split(textsplitter)
    else:
        loader = UnstructuredFileLoader(filepath, mode="elements")
        textsplitter = RecursiveCharacterTextSplitter(pdf=True, separators=["Memory:"])
        docs = loader.load_and_split(text_splitter=textsplitter)
    return docs


def load_memory_file(filepath, user_name, language="cn"):
    loader = JsonMemoryLoader(filepath, language)
    docs = loader.load(user_name)
    return docs


def get_docs_with_score(docs_with_score):
    docs = []
    for doc, score in docs_with_score:
        doc.metadata["score"] = score
        docs.append(doc)
    return docs


def seperate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists


def similarity_search_with_score_by_vector(
    self,
    embedding: List[float],
    k: int = 4,
) -> List[Tuple[Document, float]]:
    scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
    docs = []
    id_set = set()
    for j, i in enumerate(indices[0]):
        if i == -1:
            continue
        _id = self.index_to_docstore_id[i]
        doc = self.docstore.search(_id)
        id_set.add(i)
        docs_len = len(doc.page_content)
        for k in range(1, max(i, len(docs) - i)):
            for l in [i + k, i - k]:
                if 0 <= l < len(self.index_to_docstore_id):
                    _id0 = self.index_to_docstore_id[l]
                    doc0 = self.docstore.search(_id0)
                    if docs_len + len(doc0.page_content) > self.chunk_size:
                        break
                    elif doc0.metadata["source"] == doc.metadata["source"]:
                        docs_len += len(doc0.page_content)
                        id_set.add(l)
    id_list = sorted(list(id_set))
    id_lists = seperate_list(id_list)
    for id_seq in id_lists:
        for idx in id_seq:
            if idx == id_seq[0]:
                _id = self.index_to_docstore_id[idx]
                doc = self.docstore.search(_id)
            else:
                _id0 = self.index_to_docstore_id[idx]
                doc0 = self.docstore.search(_id0)
                doc.page_content += doc0.page_content
        if not isinstance(doc, Document):
            raise ValueError(f"Could not find document for id {_id}, got {doc}")
        docs.append((doc, scores[0][j]))
    return docs


class LocalMemoryRetrieval:
    embeddings: object = None
    top_k: int = VECTOR_SEARCH_TOP_K
    chunk_size: int = CHUNK_SIZE

    def init_cfg(
        self,
        embedding_model: str = EMBEDDING_MODEL_CN,
        embedding_device=EMBEDDING_DEVICE,
        top_k=VECTOR_SEARCH_TOP_K,
        language="cn",
    ):
        self.language = language
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_dict[embedding_model]
        )
        self.top_k = top_k

    def init_memory_vector_store(
        self,
        filepath: str or List[str],  # type: ignore
        vs_path: str or os.PathLike = None,  # type: ignore
        user_name: str = None,
        cur_date: str = None,
    ):
        loaded_files = []
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                print("Path does not exist")
                return None, None
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                docs = load_memory_file(filepath, user_name, self.language)
                print(f"{file} loaded successfully")
                loaded_files.append(filepath)
            elif os.path.isdir(filepath):
                docs = []
                for file in os.listdir(filepath):
                    fullfilepath = os.path.join(filepath, file)
                    try:
                        docs += load_memory_file(fullfilepath, user_name, self.language)
                        print(f"{file} loaded successfully")
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        print(e)
                        print(f"{file} failed to load")
        else:
            docs = []
            for file in filepath:
                try:
                    docs += load_memory_file(file, user_name, self.language)
                    print(f"{file} loaded successfully")
                    loaded_files.append(file)
                except Exception as e:
                    print(e)
                    print(f"{file} failed to load")
        if len(docs) > 0:
            if vs_path and os.path.isdir(vs_path):
                vector_store = FAISS.load_local(vs_path, self.embeddings)
                print(f"Load from previous memory index {vs_path}.")
                vector_store.add_documents(docs)
            else:
                if not vs_path:
                    vs_path = f"""{VS_ROOT_PATH}{os.path.splitext(file)[0]}_FAISS_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}"""
                vector_store = FAISS.from_documents(docs, self.embeddings)

            vector_store.save_local(vs_path)
            return vs_path, loaded_files
        else:
            print(
                "Files failed to load. Please check dependencies or upload a different file."
            )
            return None, loaded_files

    def load_memory_index(self, vs_path):
        vector_store = FAISS.load_local(vs_path, self.embeddings)
        FAISS.similarity_search_with_score_by_vector = similarity_search_with_score_by_vector
        vector_store.chunk_size = self.chunk_size
        return vector_store

    def search_memory(self, query, vector_store):
        related_docs_with_score = vector_store.similarity_search_with_score(
            query, k=self.top_k
        )
        related_docs = get_docs_with_score(related_docs_with_score)
        related_docs = sorted(related_docs, key=lambda x: x.metadata["source"], reverse=False)
        pre_date = ""
        date_docs = []
        dates = []
        for doc in related_docs:
            doc.page_content = doc.page_content.replace(
                f'Time {doc.metadata["source"]} conversation content:', ""
            ).strip()
            if doc.metadata["source"] != pre_date:
                date_docs.append(doc.page_content)
                pre_date = doc.metadata["source"]
                dates.append(pre_date)
            else:
                date_docs[-1] += f"\n{doc.page_content}"
        return date_docs, ", ".join(dates)
```