# -*- coding: utf-8 -*-

import sys
import os
import json
import time
import copy
import argparse
from typing import Dict, Any
from openai import OpenAI
import re
import openai

# Add project root and memory bank path safely
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'memory_bank'))

# Configuration for OpenAI API
CHATGPT_CONFIG = {
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 400,
    "top_p": 1.0,
    "frequency_penalty": 0.4,
    "presence_penalty": 0.2,
    "stop": ["<|im_end|>", "¬Human"]
}

if os.path.exists("/data"):
    BASE_DATA_DIR = "/data"
else:
    BASE_DATA_DIR = os.environ.get("EMMA_DATA_DIR", PROJECT_ROOT)

MEMORY_DIR = os.path.join(BASE_DATA_DIR, "Emma-memory-storage", "update_memory_0512_eng.json")



class LLMClientSimple:
    def __init__(self, config, gen_config=None):
        self.config = config
        self.client = OpenAI(
            api_key=os.environ.get("GAPGPT_API_KEY", "YOUR_API_KEY_HERE"),
            base_url=os.environ.get("GAPGPT_BASE_URL", "https://api.gapgpt.app/v1")
        )
        self.disable_tqdm = False

        
        self.disable_tqdm = False
        self.gen_config = copy.deepcopy(
            gen_config if gen_config is not None else config
        )

        self.disable_tqdm = False
        self.gen_config = gen_config

    def generate_text_simple(self, prompt, prompt_num, language='en'):
        self.gen_config['n'] = prompt_num
        retry_times, count = 5, 0
        response = None
        
        while response is None and count < retry_times:
            try:
                request = copy.deepcopy(self.gen_config)
                
                # System prompt setup based on language
                if language == 'cn':
                    message = [
                        {"role": "system", "content": "The following is a transcript of a conversation between a human and a smart, psychology-savvy AI assistant."},
                        {"role": "user", "content": "Hello! Please help me summarize the conversation"},
                        {"role": "system", "content": "OK, I'll try to help you."},
                        {"role": "user", "content": f"{prompt}"}
                    ]
                else:
                    message = [
                        {"role": "system", "content": "Below is a transcript of a conversation between a human and an AI assistant that is intelligent and knowledgeable in psychology."},
                        {"role": "user", "content": "Hello! Please help me summarize the content of the conversation."},
                        {"role": "system", "content": "Sure, I will do my best to assist you."},
                        {"role": "user", "content": f"{prompt}"}
                    ]

                # Call the API method
                response = self.client.chat.completions.create(**request, messages=message)

            except Exception as e:
                print(e)
                # Check for context limit errors
                if 'context' in str(e).lower() or 'maximum' in str(e).lower():
                    cut_length = 1800 - 200 * count
                    print('Max context length reached, cut to {}'.format(cut_length))
                    prompt = prompt[-cut_length:]
                    response = None
                count += 1
        
        if response:
            # Get response content using dot notation
            task_desc = response.choices[0].message.content
        else:
            task_desc = ''
            
        return task_desc


# Lazy initialization instead of module-level
_llm_client = None

def get_llm_client():
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClientSimple(CHATGPT_CONFIG, gen_config=CHATGPT_CONFIG)
    return _llm_client



# --- Prompt Generation Functions ---

def summarize_content_prompt(content, user_name, boot_name, language='en'):
    # Base prompt adjusted based on psychology
    prompt = 'Please summarize the following dialogue from a psychological perspective, focusing on the user\'s emotional state, key concerns, and potential therapeutic insights. Dialogue content:\n'
    
    for dialog in content:
        query = dialog['query']
        response = dialog['response']
        prompt += f"\n{user_name}: {query.strip()}"
        prompt += f"\n{boot_name}: {response.strip()}"
    
    prompt += ('\nSummary:' if language == 'cn' else '\nSummarization:')
    return prompt


def summarize_user_issues_prompt(content, user_name, boot_name, language='en'):
    """
    Analyze the conversation from a psychological perspective, identify user problems and their causes,
    and organize them into a structured JSON format.
    """
    prompt = (
        "Please analyze the following dialogue from a psychological perspective. Identify the user's key "
        "problems and the underlying causes behind them. Structure the results in a clear and concise way. "
        "Dialogue content:\n"
    )

    for dialog in content:
        query = dialog['query']
        response = dialog['response']
        prompt += f"\n{user_name}: {query.strip()}"
        prompt += f"\n{boot_name}: {response.strip()}"

    prompt += '\n\nPlease list the user’s problems and their causes in JSON format, like this:\n'
    prompt += '{\n  "problems": [\n    {\n      "problem": "<problem_1>",\n      "cause": "<cause_1>"\n    },\n    {\n      "problem": "<problem_2>",\n      "cause": "<cause_2>"\n    }\n  ]\n}'

    return prompt


def summarize_overall_prompt(content, language='en'):
    prompt = 'Please summarize the following events in a high-level manner, and try to be as concise as possible, summarizing and retaining the core key information:\n' if language == 'cn' else "Please provide a highly concise summary of the following event, capturing the essential key information as succinctly as possible. Summarize the event:\n"
    
    for date, summary_dict in content:
        summary = summary_dict['content']
        prompt += (f"\nTime {date} The event that occurred was {summary.strip()}" if language == 'cn' else f"At {date}, the events are {summary.strip()}")
    
    prompt += ('\nSummary:' if language == 'cn' else '\nSummarization:')
    return prompt


def summarize_overall_personality(content, language='en'):
    prompt = 'Below are the personality traits and moods of users in multiple conversations, as well as the appropriate response strategies at the time:\n' if language == 'cn' else "The following are the user's exhibited personality traits and emotions throughout multiple dialogues, along with appropriate response strategies for the current situation:"
    
    for date, summary in content:
        prompt += (f"\n in time {date} The analysis is {summary.strip()}" if language == 'cn' else f"At {date}, the analysis shows {summary.strip()}")
    
    prompt += ('\nPlease give a general summary of the users personality and the most appropriate response strategy for the AI lover, and try to be concise and highly summarized. The summary is:' if language == 'cn' else "Please provide a highly concise and general summary of the user's personality and the most appropriate response strategy for the AI lover, summarized as:")
    return prompt


def summarize_person_prompt(content, user_name, boot_name, language):
    prompt = f"Based on the following dialogue, analyze {user_name}'s emotional state, cognitive patterns, and potential psychological needs. Suggest therapeutic strategies for the AI to respond empathetically and effectively. Dialogue content:\n"
    
    for dialog in content:
        query = dialog['query']
        response = dialog['response']
        prompt += f"\n{user_name}: {query.strip()}"
        prompt += f"\n{boot_name}: {response.strip()}"

    prompt += (f'\n{user_name} personality traits, mood, {boot_name} The response strategy is:' if language == 'cn' else f"\n{user_name}'s personality traits, emotions, and {boot_name}'s response strategy are:")
    return prompt


# --- Memory Extraction Functions ---

def extract_session_summary(conversation_text, session_date, session_id):
    """
    Extracts session-level insights and creates an episodic memory entry.

    Args:
        conversation_text (str): The full conversation from the session.
        session_date (str): The date of the session.
        session_id (int): The index of the session.

    Returns:
        dict: Extracted episodic memory entry.
    """
    # Prompt to the LLM to extract insights
    insight_prompt = f"""
Please analyze the following conversation and extract key information.

Conversation:
{conversation_text}

CRITICAL INSTRUCTIONS:
1. You MUST output ONLY a valid JSON object.
2. DO NOT include any conversational text, greetings, or explanations (e.g., do not say "Sure!").
3. DO NOT include markdown formatting like
```json.
4. DO NOT include any comments (like //) inside the JSON.

Format response EXACTLY as this JSON structure:
{{
  "topics_discussed": [],
  "emotional_state": "",
  "insights": ""
}}
"""

    response = get_llm_client().generate_text_simple(
        prompt=insight_prompt,
        prompt_num=1,
        language="en",
    )

    # Remove extra whitespace
    response = response.strip()

    # حذف تگ شروع بلاک کد مارک‌داون
    if response.startswith("```json"):response = response[7:]
    elif response.startswith("```"):response = response[3:]

    # حذف تگ پایان بلاک کد مارک‌داون
    if response.endswith("```"):
        response = response[:-3]

        response = response.strip()

    try:
        episodic_memory = json.loads(response)
    except json.JSONDecodeError:
        print("⚠️ Failed to parse episodic memory response!")
        episodic_memory = {
        "topics_discussed": ["N/A"],
        "emotional_state": "unknown",
        "insights": "No insights extracted.",
        }

    # Add session metadata
    episodic_memory["session_date"] = session_date
    episodic_memory["session_id"] = session_id

    return episodic_memory



def extract_semantic_memory(latest_episodic_memory, existing_semantic_memory):
    """
    Updates semantic memory based on the latest episodic memory.
    """
    # Use the insights from episodic memory to infer stable traits and patterns
    semantic_prompt = f"""
Please analyze the following session summary and infer long-term personality traits and stable characteristics:
Session Summary:
{json.dumps(latest_episodic_memory, indent=4)}

CRITICAL INSTRUCTIONS:
1. You MUST output ONLY a valid JSON object.
2. DO NOT include any conversational text, greetings, or explanations.
3. DO NOT include markdown formatting like 
```json.
    4. DO NOT include any comments inside the JSON.

    Update the user's semantic memory with:
    1. Evolving personality traits (Big Five).
    2. Core values and recurring motivations.
    3. Behavioral patterns and consistent emotional responses.
    4. Frequent topics and recurring themes.

    Format response EXACTLY as this JSON structure:
    {{
    "personality_traits": {{
    "openness": 0.0,
    "conscientiousness": 0.0,
    "extraversion": 0.0,
    "agreeableness": 0.0,
    "neuroticism": 0.0
    }},
    "core_values": [],
    "behavioral_patterns": [],
    "recurring_themes": []
    }}
    """

    response = get_llm_client().generate_text_simple(prompt=semantic_prompt, prompt_num=1, language='en')

    # Safe JSON extraction using Regex
    try:
        # Find content between first { and last }
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            new_semantic_data = json.loads(json_str)
        else:
            raise ValueError("No JSON object found in response.")

    except Exception as e:
        print(f"⚠️ Failed to parse semantic memory response! Error: {e}")
        print(f"LLM Response was:\n{response}") 
        return existing_semantic_memory

    # Merge new data into semantic memory
    updated_semantic_memory = existing_semantic_memory if isinstance(existing_semantic_memory, dict) else {}

    # Update or average personality traits
    for trait, value in new_semantic_data.get("personality_traits", {}).items():
        try:
            val_float = float(value)
        except (ValueError, TypeError):
            val_float = 0.0

        if trait in updated_semantic_memory.get("personality_traits", {}):
            existing_value = float(updated_semantic_memory["personality_traits"].get(trait, 0.0))
            updated_semantic_memory["personality_traits"][trait] = round((existing_value + val_float) / 2, 2)
        else:
            updated_semantic_memory.setdefault("personality_traits", {})[trait] = val_float

    # Combine core values, avoiding duplicates
    updated_semantic_memory["core_values"] = list(set(
        updated_semantic_memory.get("core_values", []) + new_semantic_data.get("core_values", [])
    ))

    # Combine behavioral patterns
    updated_semantic_memory["behavioral_patterns"] = list(set(
        updated_semantic_memory.get("behavioral_patterns", []) + new_semantic_data.get("behavioral_patterns", [])
    ))

    
    updated_semantic_memory["recurring_themes"] = list(set(
        updated_semantic_memory.get("recurring_themes", []) + new_semantic_data.get("recurring_themes", [])
    ))

    print("\n✅ Semantic memory updated successfully!")
    return updated_semantic_memory

    
    updated_semantic_memory = existing_semantic_memory if isinstance(existing_semantic_memory, dict) else {}

    
    for trait, value in new_semantic_data.get("personality_traits", {}).items():
        try:
            val_float = float(value)
        except (ValueError, TypeError):
            val_float = 0.0

        if trait in updated_semantic_memory.get("personality_traits", {}):
            existing_value = float(updated_semantic_memory["personality_traits"].get(trait, 0.0))
            updated_semantic_memory["personality_traits"][trait] = round((existing_value + val_float) / 2, 2)
        else:
            updated_semantic_memory.setdefault("personality_traits", {})[trait] = val_float

    # Combine core values, avoiding duplicates
    updated_semantic_memory["core_values"] = list(set(
        updated_semantic_memory.get("core_values", []) + new_semantic_data.get("core_values", [])
    ))

    # Combine behavioral patterns
    updated_semantic_memory["behavioral_patterns"] = list(set(
        updated_semantic_memory.get("behavioral_patterns", []) + new_semantic_data.get("behavioral_patterns", [])
    ))

    # Combine recurring themes
    updated_semantic_memory["recurring_themes"] = list(set(
        updated_semantic_memory.get("recurring_themes", []) + new_semantic_data.get("recurring_themes", [])
    ))

    print("\n✅ Semantic memory updated successfully!")
    return updated_semantic_memory




def summarize_memory(memory_dir, name=None, language='en'):
    boot_name = 'AI'
    gen_prompt_num = 1

    with open(memory_dir, 'r', encoding='utf8') as f:
        memory = json.loads(f.read())

    for k, v in memory.items():
        if name is not None and k != name:
            continue

        user_name = k
        print(f'Updating memory for user {user_name}')

        if v.get('history') is None:
            continue

        history = v['history']

        if v.get('summary') is None:
            memory[user_name]['summary'] = {}
        if v.get('personality') is None:
            memory[user_name]['personality'] = {}
        if v.get('issues') is None:
            memory[user_name]['issues'] = {}

        for date, content in history.items():
            # Check flags to see if update is needed
            his_flag = False if (date in v['summary'].keys() and v['summary'][date]) else True
            person_flag = False if (date in v['personality'].keys() and v['personality'][date]) else True
            problem_flag = False if (date in v['issues'].keys() and v['issues'][date]) else True 

            hisprompt = summarize_content_prompt(content, user_name, boot_name, language)
            issues_prompt = summarize_user_issues_prompt(content, user_name, boot_name, language)
            person_prompt = summarize_person_prompt(content, user_name, boot_name, language)

            if his_flag:
                his_summary = get_llm_client().generate_text_simple(prompt=hisprompt, prompt_num=gen_prompt_num, language=language)
                memory[user_name]['summary'][date] = {'content': his_summary}

            if problem_flag:
                issues_summary = get_llm_client().generate_text_simple(prompt=issues_prompt, prompt_num=gen_prompt_num, language=language)
                memory[user_name]['issues'][date] = {'content': issues_summary}
                print(issues_summary)

            if person_flag:
                person_summary = get_llm_client().generate_text_simple(prompt=person_prompt, prompt_num=gen_prompt_num, language=language)
                memory[user_name]['personality'][date] = person_summary

        overall_his_prompt = summarize_overall_prompt(list(memory[user_name]['summary'].items()), language=language)
        overall_person_prompt = summarize_overall_personality(list(memory[user_name]['personality'].items()), language=language)

        memory[user_name]['overall_history'] = get_llm_client().generate_text_simple(prompt=overall_his_prompt, prompt_num=gen_prompt_num, language=language)
        memory[user_name]['overall_personality'] = get_llm_client().generate_text_simple(prompt=overall_person_prompt, prompt_num=gen_prompt_num, language=language)
     
    with open(memory_dir, 'w', encoding='utf8') as f:
        print(f'Successfully updated memory for {name}')
        json.dump(memory, f, ensure_ascii=False)
        
    return memory


if __name__ == '__main__':
    # مسیردهی هوشمند برای محیط ابری
    base_dir = "/data" if os.path.exists("/data") else os.environ.get("EMMA_DATA_DIR", PROJECT_ROOT)
    target_memory = os.path.join(base_dir, 'Emma-memory-storage', 'eng_memory_cases.json')

    
    if os.path.exists(target_memory):
        summarize_memory(target_memory, language='en')
    else:
        print(f"File not found: {target_memory}")


