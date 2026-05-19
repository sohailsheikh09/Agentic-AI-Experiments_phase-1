# Agentic AI Experiments — Phase 1

Agentic AI experiments using **LangGraph** — covers human-in-the-loop workflows, RAG agents, sentiment analysis, LinkedIn post generation, chatbots with memory, and more.

---

## Experiments

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `001_simple_llm_workflow.ipynb` | Basic LLM chain workflow using LangGraph |
| 2 | `002_sentiment_review_reply_workflow.ipynb` | Sentiment analysis + automated review reply generation |
| 3 | `003_simple_rag_agent.ipynb` | Simple Retrieval-Augmented Generation (RAG) agent |
| 4 | `4_X_post_generator_workflow.ipynb` | X (Twitter) post generator agentic workflow |
| 5 | `5_Chatbot_with_InMemory_Saver.ipynb` | Chatbot with in-memory conversation history |
| 6 | `6_sqlite_saver.py` | Persistent conversation state using SQLite checkpointer |
| 7 | `7_persistance.ipynb` | State persistence across sessions in LangGraph |
| 8 | `8_tools.ipynb` | LangGraph agent with tool calling capabilities |
| 9 | `9_HITL.ipynb` | Human-in-the-Loop (HITL) LinkedIn post generator with approval workflow |
| 10 | `10_map_reduce.ipynb` | Map-reduce pattern for parallel LLM processing |

### Bonus
- `chatapp_assignment/app.py` — Streamlit chat application

---

## Key Concepts Covered

- **LangGraph** — Building stateful agentic workflows as directed graphs
- **Human-in-the-Loop (HITL)** — Pausing workflows for human review using `interrupt()` and `Command(resume=...)`
- **State Persistence** — `InMemorySaver` and `SQLiteSaver` checkpointers
- **RAG** — Retrieval-Augmented Generation for grounded responses
- **Tool Calling** — Integrating external tools into LLM agents
- **Map-Reduce** — Parallel processing patterns in agentic pipelines

---

## Setup

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (package manager)

### Install dependencies
```bash
uv sync
```

### Environment variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key        # if used
LINKEDIN_ACCESS_TOKEN=your_linkedin_token  # for HITL notebook
LINKEDIN_AUTHOR_URN=urn:li:person:xxxxxx   # for HITL notebook
```

### Run notebooks
Open any `.ipynb` file in VS Code or JupyterLab and run cells sequentially.

---

## Tech Stack

- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [LangChain](https://www.langchain.com/)
- [Groq](https://groq.com/) (LLM provider)
- [Streamlit](https://streamlit.io/) (chat UI)
- Python `sqlite3` (persistence)
