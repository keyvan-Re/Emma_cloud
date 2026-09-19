import openai
import os
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

GAPGPT_API_KEY = os.environ.get("GAPGPT_API_KEY", "your_default_api_key_here")

Settings.llm = OpenAI(api_key="GAPGPT_API_KEY", api_base="https://api.gapgpt.app/v1", model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(api_key="GAPGPT_API_KEY", api_base="https://api.gapgpt.app/v1", model="text-embedding-ada-002")

boot_name_dict = {'en':'AI Companion'}
boot_actual_name_dict = {'en':'Emma'}
def output_prompt(history, user_name, boot_name):
    prompt = f""
    for dialog in history:
        if isinstance(dialog, dict):
            query = dialog.get('query', '')
            response = dialog.get('response', '')
        else:
            query, response = dialog
        prompt += f"\n\n{user_name}：{query}"
        prompt += f"\n\n{boot_name}：{response}"
    return prompt

def generate_meta_prompt_dict_chatgpt():
    meta_prompt_dict = {'cn':"""
    You  play the role of an AI assistant in the field of psychology for this user ({user_name}). 
        Your goal is to provide emotionally supportive, scientifically grounded, and empathetic responses. 
        Use the following user-specific details to inform your response:

        - **User's Psychological Profile:** {}
        - **Summary of Past Interactions:** {}
        - **Relevant Past Conversations:** {related_memory_content} 
        
        The user has asked: 

        Please provide an insightful and appropriate response considering their personal history.
    """,
    'en':"""
    You  play the role of an AI assistant in the field of psychology for this user ({user_name}). 
        Your goal is to provide emotionally supportive, scientifically grounded, and empathetic responses. 
        Use the following user-specific details to inform your response:

        - **User's Psychological Profile:** {personality}
        - **Summary of Past Interactions:** {history_summary}
        - **Relevant Past Conversations:** {related_memory_content} 
        
        The user has asked: 

        Please provide an insightful and appropriate response considering their personal history.
    """,} 
    return meta_prompt_dict
def generate_meta_prompt_dict_semantic_chatgpt():
    meta_prompt_dict = {'cn':"""  
    """,

    'en':"""
    
You play the role of an AI assistant in the field of psychology for this user ({user_name}). 
Your goal is to provide emotionally supportive, scientifically grounded, and empathetic responses. 
Use the following user-specific semantic memory{semantic_memory_text} to tailor your response:
    """,} 
    print(meta_prompt_dict)  
    return meta_prompt_dict

def generate_meta_prompt_dict_semantic_episodic_chatgpt():
    meta_prompt_dict = {'cn':"""
    You  play the role of an AI assistant in the field of psychology for this user ({user_name}). 
        Your goal is to provide emotionally supportive, scientifically grounded, and empathetic responses. 
        Use the following user-specific details to inform your response:

        - **User's Psychological Profile:** {}
        - **Summary of Past Interactions:** {}
        - **Relevant Past Conversations:** {related_memory_content} 
        
        The user has asked: 

        Please provide an insightful and appropriate response considering their personal history.
    """,

    'en': """
You are an AI assistant specializing in psychology, assisting the user ({user_name}) in a supportive, scientifically grounded, and empathetic manner.

Use the following memory sections to personalize your response:

- **User's Long-Term Traits and Psychological Characteristics (Semantic Memory):**
{semantic_memory_text}

- **Important Past Interactions or Events (Episodic Memory):**
{related_memory_content}

Respond by considering both the long-term traits and recent events of the user, offering personalized, emotionally intelligent support. Keep your response helpful, psychologically informed, and sensitive to the user's unique situation.
""",} 
    return meta_prompt_dict

def generate_new_user_meta_prompt_dict_chatgpt():
    meta_prompt_dict = {'cn':"""
    Now you will play the role of an companion AI Companion for user {user_name}, and your name is {boot_actual_name}. You should be able to: (1) provide warm companionship to chat users; (2) you are also an excellent psychological counselor, and when users confide in you about their difficulties and seek help, you can provide them with warm and helpful responses.
    """,
    'en':"""
    Now you will play the role of an companion AI Companion for user {user_name}, and your name is {boot_actual_name}. You should be able to: (1) provide warm companionship to chat users; (2) you are also an excellent psychological counselor, and when users confide in you about their difficulties and seek help, you can provide them with warm and helpful responses.
    """} 
    return meta_prompt_dict

def generate_user_keyword():
    return {'cn': '[|User|]', 'en': '[|User|]'}

def generate_ai_keyword():
    return {'cn': '[|AI|]', 'en': '[|AI|]'}


def build_prompt_with_search_memory_llamaindex(
    history, 
    query,  
    user_memory, 
    user_name, 
    user_memory_index, 
    service_context, 
    api_keys, 
    api_index, 
    meta_prompt, 
    new_user_meta_prompt, 
    data_args, 
    boot_actual_name,
    semantic_memory_text,
    query_category,
    meta_prompt_semantic,
    meta_prompt_semantic_episodic
):
  
    memory_search_query = query
    related_memos = ""
    
    print("memory_search_query:", memory_search_query)
   #print("Total nodes in index:", len(user_memory_index.docstore.docs))

    if user_memory_index:
        retried_times = 2 
        count = 0
        
        while not related_memos and count < retried_times:
            try:
              
                retriever = user_memory_index.as_retriever(
                    similarity_top_k=3,  
                )
                
             
                nodes = retriever.retrieve(memory_search_query)
                
                if nodes:
                    related_memos = "\n\n".join([n.node.text for n in nodes])
                else:
                    related_memos = ""
                
            except Exception as e:
                print(f"Error querying index (attempt {count+1}/{retried_times}): {e}")
                
                try:
                    api_index = (api_index + 1) % len(api_keys)
                    openai.api_key = api_keys[api_index] 
                except:
                    pass
                
                related_memos = ""
                
            count += 1

    # Process history
    print("related_memos found:", len(related_memos) if related_memos else 0)

    #-********************Debug*********************
    print("[MEMORY DEBUG] category:", query_category, flush=True)
    print(
        "[MEMORY DEBUG] result type:",
        type(related_memos).__name__,
        flush=True,
    )
    print("[MEMORY DEBUG] retrieved content BEGIN", flush=True)
    print(related_memos, flush=True)
    print("[MEMORY DEBUG] retrieved content END", flush=True)

     #-********************Debug*********************
    
    history_summary = ""
    if "overall_history" in user_memory:
        history_summary = f"The summary of your past memories with the user is: {user_memory['overall_history']}"
    
    personality = user_memory.get("overall_personality", "")
    print("query_category:", query_category)

    # Construct the prompt
    if query_category == "semantic_memory":
        prompt = meta_prompt_semantic.format(
            user_name=user_name,
            semantic_memory_text=semantic_memory_text
        )
        
    elif query_category == "semantic-episodic":
        prompt = meta_prompt_semantic_episodic.format(
            user_name=user_name,
            history_summary=history_summary,
            related_memory_content=f"\n{str(related_memos).strip()}\n" if related_memos else "None",
            personality=personality,
            boot_actual_name=boot_actual_name,
            semantic_memory_text=semantic_memory_text
        )
        
    elif query_category == "episodic_memory":
        if related_memos:
            prompt = meta_prompt.format(
                user_name=user_name,
                history_summary=history_summary,
                related_memory_content=f"\n{str(related_memos).strip()}\n",
                personality=personality,
                boot_actual_name=boot_actual_name
            )
        else:
            prompt = new_user_meta_prompt.format(
                user_name=user_name,
                boot_actual_name=boot_actual_name
            )
            
    else: 
        prompt = new_user_meta_prompt.format(
            user_name=user_name,
            boot_actual_name=boot_actual_name
        )
    
    return prompt, related_memos
