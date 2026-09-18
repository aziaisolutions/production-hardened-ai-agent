# Enterprise AI Support & Booking Agent

A production-hardened LLM orchestration pipeline built with Streamlit, LangChain, and Groq (LLaMA-3).

## Solved Vulnerabilities
- **Pre-Retrieval PII Sanitization (Issue #05)**: In-memory regex masking of Phone, Email, and National IDs before LLM ingestion.
- **State-Level Circuit Breakers (Issue #01)**: Hard-capped execution retries to eliminate recursive token loops.

## Tech Stack
Python, Streamlit, LangChain, Groq
