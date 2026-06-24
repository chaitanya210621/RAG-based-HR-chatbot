# RAG-Based HR Policy Chatbot

## Overview

This project presents a Retrieval-Augmented Generation (RAG) based HR Policy Query System designed to provide accurate, context-aware answers to employee HR-related questions. Traditional keyword-based search systems often fail to understand user intent and provide relevant responses. This system combines semantic retrieval and Large Language Models (LLMs) to deliver reliable answers grounded in organizational HR policy documents.

The application processes HR policy documents, converts them into vector embeddings, retrieves the most relevant information using semantic search, and generates natural language responses with source-backed explanations.

---

## Problem Statement

Organizations maintain extensive HR policy documents covering attendance, leave management, salaries, employee conduct, travel allowances, and performance evaluation. Employees often struggle to locate relevant information quickly using traditional search methods.

This project addresses the challenge by building an intelligent HR assistant capable of understanding natural language queries and retrieving precise policy information with high contextual accuracy.

---

## Key Features

* HR policy document-based question answering
* Retrieval-Augmented Generation (RAG) architecture
* Semantic document chunking
* Vector similarity search using FAISS
* Context-aware response generation
* Source-cited answers
* Interactive Streamlit user interface
* Performance evaluation using retrieval and generation metrics

---

## System Architecture
<img width="723" height="445" alt="image" src="https://github.com/user-attachments/assets/dd6281f0-e471-45ba-b123-d5be1c6bdd5d" />


## Technologies Used

### Frontend

* Streamlit

### Backend

* Python

### AI & NLP

* LangChain
* Retrieval-Augmented Generation (RAG)
* Sentence Transformers

### Embedding Model

* all-MiniLM-L6-v2

### Vector Database

* FAISS (Facebook AI Similarity Search)

### Evaluation

* ROUGE-1
* ROUGE-2
* ROUGE-L
* Semantic Similarity
* Contextual Grounding
* Precision@3
* Mean Reciprocal Rank (MRR)

---

## Methodology

### 1. Data Collection and Preprocessing

HR policy documents covering attendance, leave policies, salary information, employee conduct, grievances, and performance evaluation are collected and cleaned for processing.

### 2. Semantic Chunking

Documents are divided into overlapping semantic chunks using recursive text splitting techniques to preserve contextual continuity.

### 3. Embedding Generation

Each document chunk is converted into dense vector representations using the all-MiniLM-L6-v2 sentence transformer model.

### 4. Vector Storage

Generated embeddings are stored in a FAISS vector database for efficient similarity search and retrieval.

### 5. Query Retrieval

User queries are embedded using the same embedding model and matched against stored vectors using cosine similarity.

### 6. Response Generation

The retrieved context is supplied to a Large Language Model through LangChain, enabling generation of accurate and grounded responses.

---

## Experimental Results

The proposed RAG system significantly outperformed a baseline Large Language Model without external retrieval.

| Metric               | RAG System | Baseline LLM | Improvement |
| -------------------- | ---------- | ------------ | ----------- |
| ROUGE-1              | 0.6766     | 0.3694       | +30.7%      |
| ROUGE-2              | 0.4305     | 0.1019       | +32.9%      |
| ROUGE-L              | 0.5932     | 0.2534       | +33.9%      |
| Semantic Similarity  | 0.9021     | 0.7441       | +15.8%      |
| Contextual Grounding | 0.7854     | 0.6509       | +13.4%      |
| Precision@3          | 0.8611     | -            | High        |
| MRR                  | 1.0        | -            | Excellent   |

---

## Key Findings

* Significant improvement in lexical and semantic accuracy.
* Reduced hallucinations through retrieval-based grounding.
* High retrieval effectiveness with Precision@3 of 0.86.
* Mean Reciprocal Rank (MRR) of 1.0 indicates relevant documents are consistently retrieved at the top rank.
* Improved trustworthiness through source-backed responses.

---

## Future Enhancements

* Hybrid dense and sparse retrieval techniques
* Advanced embedding models such as BGE-M3 and Nomic Embeddings
* Multilingual HR support
* Explainable AI-based answer tracing
* Confidence score generation
* Continuous learning through user feedback

architecture improves factual accuracy, contextual relevance, and transparency, making it suitable for real-world HR policy assistance applications.

