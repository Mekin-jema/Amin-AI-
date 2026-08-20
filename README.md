# Production LLM and RAG API

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![FastAPI: 0.110+](https://img.shields.io/badge/FastAPI-0.110%2B-green.svg)](https://fastapi.tiangolo.com/)

A production-grade, asynchronous API scaffold for operating stateful LLM agents with Retrieval-Augmented Generation (RAG). Built using **FastAPI**, **LangChain**, and **LangGraph**, this repository provides a reference implementation for complex agent workflows, external tool calling, persistent storage with **ChromaDB**, and real-time monitoring.

---

## Architecture & Features

This project implements a complete, real-world Agentic AI backend API system, graph-based agent state machines, and vector databases.

*   **Stateful Agent Workflows:** Complex agent orchestration via **LangGraph**, supporting cyclic transitions, tool execution state, and agent history.
*   **Tool Calling & Execution:** Declarative tool definitions integrated directly into the agent decision loops.
*   **Advanced RAG Patterns:** Retrieval-Augmented Generation leveraging **ChromaDB** for vector storage and semantic lookup.
*   **Persistence & Memory:** Long-term conversation memory and thread persistence.
*   **Human-in-the-Loop:** Support for human approval steps during critical agent actions.
*   **Observability:** Performance monitoring and evaluation integrations with **LangSmith**.
*   **High-Performance API Backend:** Asynchronous endpoints built on **FastAPI** and served with **Uvicorn**.

---

## Technology Stack

*   **Core Logic:** LangChain, LangGraph (Python)
*   **API Framework:** FastAPI, Pydantic, Uvicorn
*   **Vector Search & Database:** ChromaDB, PostgreSQL
*   **Deployment:** Docker & CI/CD Pipelines

---

## Getting Started

### Prerequisites

*   Python 3.12 or later
*   ChromaDB / PostgreSQL (or running instances)

### 1. Installation

Install project dependencies using the `uv` package manager:

```bash
uv sync
```

### 2. Environment Configuration

Create a `.env` file in the root directory and configure your credentials:

```env
OPENAI_API_KEY=sk-proj-...
TAVILY_API_KEY=tvly-...
```

### 3. Run the Backend API

Start the FastAPI local development server:

```bash
uv run uvicorn main:app --reload
```

---

## Project Structure

*   `app/` — Backend logic, FastAPI app, and LangGraph agent workflow definition.
*   `pyproject.toml` — Python package dependencies and metadata.
*   `README.md` — Project documentation.
