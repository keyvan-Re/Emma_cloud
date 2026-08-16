# EMMA — Empathetic Memory-Augmented Multi-layer Assistant  
*(Research Prototype)*

**Empathetic, privacy-aware memory for psychologically informed conversational agents.**

This repository provides a reference implementation of a mobile-friendly, memory-augmented artificial intelligence assistant inspired by the **EMMA architecture**. The system integrates **session**, **episodic**, and **semantic** memory layers, dynamic query classification, **LlamaIndex-based retrieval**, and a **Gradio** demonstration interface. The codebase includes implementation logic, data processing scripts, memory indexing and retrieval components, query classification, and evaluation tooling used in the research prototype.

> ⚠️ **Research prototype — not a clinical tool.**  
> This system is intended solely for research, experimentation, and controlled simulations. It must not be used for clinical diagnosis or treatment. See **Limitations & Safety** below.

---

## Table of Contents
- [Key Features](#key-features)
- [Architecture (High Level)](#architecture-high-level)
- [Personalized Response Generation Workflow](#Personalized-Response-Generation-Workflow)
- [Evaluation & Metrics](#evaluation--metrics)
- [Limitations & Safety](#limitations--safety)


---

## Key Features

- **Three-tier memory architecture**:  
  Session memory (raw conversational transcripts), episodic memory (session summaries), and semantic memory (long-term traits and values).

- **Dynamic query classifier**:  
  Routes user queries to *Episodic*, *Semantic*, *Hybrid*, or *Unrelated* processing pipelines.

- **Privacy-aware retrieval**:  
  Uses **LlamaIndex** and vector-based indexing for efficient local semantic search.

- **Therapy-aligned prompt templates**:  
  Combines retrieved memory with therapist-inspired prompt scaffolding to ensure emotional alignment.

- **Gradio-based demo interface**:  
  Lightweight chat UI with access to memory summaries and session history.

- **Evaluation tooling**:  
  Scripts supporting quantitative memory retrieval accuracy and qualitative Likert-scale evaluation pipelines.



---

## Architecture (High Level)



- **Indexing**: Episodic and semantic memory items are embedded and stored in vector indexes.
- **Routing**: A classifier determines which memory layer(s) should be queried. Hybrid queries may combine episodic and semantic retrieval.
- **Prompting**: Retrieved memory is merged into therapy-aware prompt templates prior to LLM invocation.

---

## Personalized Response Generation Workflow

The figure illustrates the end-to-end workflow used by **EMMA** for generating personalized and psychologically informed responses. The pipeline consists of six sequential stages:
![Personalized Response Generation Workflow](https://github.com/keyvan-Re/Emma/blob/8730850f8282c120b274472b84cd411a2058c2b4/assets/emma_workflow.jpg)

### Step 1 – User Query
The interaction begins when the user submits a query, which may express a psychological concern, emotional state, or a general question.

### Step 2 – Query Classification
The user query, combined with a task-specific prompt, is passed to a language model (e.g., GPT-3.5) acting as a query recognition mechanism. The query is classified into one of the following categories:

- **Episodic**: Past experiences or events  
- **Semantic**: Stable traits, preferences, or beliefs  
- **Hybrid**: Requires both episodic and semantic context  
- **Unrelated**: No memory retrieval required

### Step 3 – Memory Routing
Based on the predicted memory type, the system determines which memory layer(s) should be accessed and forwards the query together with the memory label to the retrieval module.

### Step 4 – Memory Retrieval
EMMA leverages **LlamaIndex** to retrieve relevant memory chunks from its structured memory store, which includes:

- **Session memory** (short-term conversational context)
- **Episodic memory** (summarized past interactions)
- **Semantic memory** (long-term psychological attributes and behavioral patterns)

### Step 5 – Prompt Composition
Retrieved memory content is merged with the user query using task-specific prompt templates designed to preserve emotional tone, maintain psychological coherence, and align responses with empathic counseling principles.

### Step 6 – Response Generation
The composed prompt is forwarded to the language model (e.g., GPT-3.5), which generates a personalized, memory-informed, and emotionally aligned response.  
Optionally, a post-processing module may refine tone and safety to ensure therapeutic appropriateness.

**Privacy Note:**  
Since psychologically relevant information is abstracted into episodic and semantic memory, raw session transcripts can be periodically discarded. This reduces storage overhead while enhancing user privacy and data security.

---

## Evaluation & Metrics

The original prototype evaluation included:

- **Qualitative evaluation**:  
  90 prompts rated on a 5-point Likert scale across *Personalization*, *Continuity*, and *Empathy* dimensions.

- **Quantitative evaluation**:  
  Memory retrieval accuracy computed as the normalized mean of 5-point Likert scores.

- **Automatic metrics**:  
  Automated rubric-based assessment using a stronger LLM evaluator.

### Reproducibility
Evaluation can be reproduced by preparing:
- A test set of prompts linked to memory entries.
- Scripts comparing generated responses with ground-truth memory and computing Likert-aligned scores.

---

## Limitations & Safety

- This system is **not a clinical or diagnostic tool** and must not replace licensed mental health professionals.
- Automated evaluators and LLM judgments may be noisy or biased; safety-critical use cases require clinician oversight and human-in-the-loop validation.
- The system may occasionally hallucinate memory-grounded facts; retrieval traces should always be logged for auditing and debugging.
- See the associated paper for a detailed discussion of limitations and evaluation methodology.

---




