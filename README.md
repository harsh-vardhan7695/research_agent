# Deep Research Agent

A web research agent that conducts deep research using web search, extracts relevant content, and generates cited answers. Built from scratch in Python without using LangChain, CrewAI, or similar frameworks.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)


---

## Design Note

### Target Users and Problem

**Target users:** Knowledge workers, researchers, students, and developers who need quick, well-sourced answers to complex questions without manually searching and synthesizing multiple sources.

**Problem being solved:** Web research is tedious. You search, open multiple tabs, read through content, mentally synthesize, and lose track of sources. This agent automates that workflow while maintaining citation integrity.

### Definition of "Deep Research"

For this implementation, "deep research" means:

1. **Multi-query exploration** - Not just one search, but multiple strategically chosen queries to cover different angles
2. **Source diversity** - Consulting multiple sources rather than relying on one
3. **Content analysis** - Actually reading and understanding page content, not just snippets
4. **Conflict awareness** - Detecting when sources disagree and explicitly noting it
5. **Citation integrity** - Every claim traceable to a specific source

What it's NOT: It's not meant to replace academic literature review or investigative journalism. It's for quick-turnaround factual research.

### Success Metrics

I chose these 5 metrics because they capture different dimensions of research quality:

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| **Concept Coverage** | % of expected key concepts present in answer | Measures completeness - did we actually answer the question? |
| **Citation Rate** | % of answers with source citations | Grounding - are claims backed up? |
| **Source Diversity** | Average unique domains per answer | Reduces single-source bias |
| **Uncertainty Expression** | Does agent acknowledge when evidence is weak? | Intellectual honesty - don't pretend to know what you don't |
| **Conflict Handling** | Does agent note when sources disagree? | Nuance - real research often has conflicting info |

### Data Flow and Components

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│  1. PLANNER      │  Generate research plan + search queries
│  (llm.py)        │  
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  2. SEARCH       │  Execute queries via Tavily API
│  (search.py)     │  Returns: title, url, snippet, relevance
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  3. FETCHER      │  Download full page content
│  (fetcher.py)    │  Extract readable text, handle errors
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  4. CONTEXT      │  Select best snippets for LLM
│  (context.py)    │  Score by relevance, enforce diversity
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  5. GENERATOR    │  Create cited answer via Gemini
│  (llm.py)        │  Stream response, handle conflicts
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  6. SESSION      │  Store conversation + turn history
│  (session.py)    │  Persist to JSON files
└──────────────────┘
```

**Why this architecture:**
- **Explicit stages** - Each step is isolated and testable
- **No hidden magic** - You can trace exactly what happened for any query
- **Easy to modify** - Swap out Tavily for another search API? Just change search.py

### Risks and Limitations

| Risk | Impact | Current Mitigation |
|------|--------|-------------------|
| **Rate limits** | Search/LLM APIs have quotas | Caching, request batching, graceful degradation |
| **Low-quality sources** | Bad sources = bad answers | Position-based scoring (trust search ranking), domain diversity |
| **Paywalled content** | Can't access full articles | Detect blocked pages, fall back to snippets |
| **Context length** | Can't fit everything | Chunk scoring + selection, summarization fallback |
| **Conflicting sources** | Sources disagree | Explicit conflict detection and disclosure |
| **Stale information** | Web search may return outdated content | Prefer recent results, note when info may be dated |

### Future Improvements

**1. Semantic retrieval for context selection**

Currently using keyword overlap to score chunks. This misses semantically similar content that uses different words. Adding embedding-based similarity (using a local model like `sentence-transformers`) would significantly improve context quality.

Implementation sketch:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good quality
query_emb = model.encode(query)
chunk_emb = model.encode(chunks)
scores = cosine_similarity(query_emb, chunk_emb)
```

**2. Iterative research with self-critique**

Current flow is single-pass. Better approach: after generating initial answer, have the LLM critique it for gaps, then do targeted follow-up searches. This catches cases where the first search didn't find key information.

---

## Setup Instructions

### Prerequisites

- Python 3.9+
- API keys for Tavily and Google AI

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd deep_research_agent

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API keys
export TAVILY_API_KEY="your-tavily-key"     # Get from https://tavily.com
export GOOGLE_API_KEY="your-google-key"     # Get from https://aistudio.google.com/apikey
```

### Running the App

```bash
# Start Streamlit interface
streamlit run app.py

# Or run CLI test
python -m core.agent
```

---

## Project Structure

```
deep_research_agent/
├── app.py                 # Streamlit UI
├── requirements.txt       # Dependencies
├── README.md             # This file
├── core/
│   ├── __init__.py
│   ├── agent.py          # Main orchestration
│   ├── search.py         # Tavily API integration
│   ├── fetcher.py        # Page fetching + extraction
│   ├── context.py        # Context building
│   ├── llm.py            # Gemini integration
│   └── session.py        # Session management
├── evaluation/
│   ├── __init__.py
│   ├── dataset.json      # Test questions
│   └── evaluator.py      # Evaluation harness
└── data/
    └── sessions/         # Stored session files
```

---

## Example Conversations

### Example 1: Factual Question

**User:** What is retrieval augmented generation and how does it work?

**Agent Response:**

> 🔍 Planning research approach...
>
> 📋 Research plan:
> UNDERSTANDING: User wants to know what RAG is and its mechanism
> SEARCHES:
> - retrieval augmented generation explained
> - how does RAG work in LLMs
> - RAG architecture components
>
> 🌐 Searching the web with 3 queries...
> 📊 Found 12 unique results
> 📥 Fetching content from top sources...
> ✅ Successfully fetched 4 sources
> 📝 Analyzing and selecting relevant content...
> 
> ---
>
> Retrieval Augmented Generation (RAG) is a technique that enhances large language models by combining retrieval from external knowledge bases with text generation [Understanding RAG — aws.amazon.com](https://aws.amazon.com/...).
>
> Here's how it works:
> 1. **Query encoding**: The user's question is converted to a vector embedding
> 2. **Retrieval**: Similar documents are retrieved from a vector database
> 3. **Context injection**: Retrieved documents are added to the LLM prompt
> 4. **Generation**: The LLM generates a response using the retrieved context
>
> [IBM RAG Guide — ibm.com](https://www.ibm.com/...)

### Example 2: Comparison Question

**User:** Compare vector databases: Pinecone vs Milvus vs Chroma

**Agent Response:** *(Shows research planning, searching, then...)*

> Here's a comparison of three popular vector databases:
>
> | Feature | Pinecone | Milvus | Chroma |
> |---------|----------|--------|--------|
> | Deployment | Managed cloud | Self-hosted/cloud | Local/embedded |
> | Scale | High (billions) | High | Low-medium |
> | Cost | Per-query pricing | Free (infra costs) | Free |
> | Best for | Production at scale | Full control needed | Prototyping, dev |
>
> **Pinecone** is fully managed, meaning you don't handle infrastructure [Pinecone Docs — pinecone.io](https://pinecone.io/...)
>
> **Milvus** offers more control and is open-source, better for organizations with specific compliance needs [Milvus Overview — milvus.io](https://milvus.io/...)
>
> **Chroma** is lightweight and runs locally, ideal for development and small projects [Chroma Getting Started — trychroma.com](https://trychroma.com/...)

### Example 3: Handling Insufficient Evidence

**User:** What is the internal architecture of Claude's neural network?

**Agent Response:**

> Based on my research, I couldn't find detailed public information about Claude's internal architecture. Anthropic has not published the specific architectural details of Claude's neural network [Anthropic Research — anthropic.com](https://anthropic.com/research).
>
> What is publicly known:
> - Claude is a large language model built by Anthropic
> - It uses transformer architecture (like most modern LLMs)
> - Anthropic emphasizes "Constitutional AI" training methods
>
> For specific architecture details (parameter count, layer structure, etc.), you would need access to Anthropic's internal documentation which is not publicly available.

---

## Evaluation Methodology

### Approach

Rather than optimizing for a single "accuracy" number (which is reductive), the evaluation measures multiple dimensions of research quality:

1. **Grounding metrics** - Is the output traceable to sources?
2. **Correctness metrics** - Does it contain the right information?
3. **Robustness metrics** - How does it handle edge cases?

### Dataset Design

The evaluation dataset (`evaluation/dataset.json`) includes:

| Category | Purpose | Example |
|----------|---------|---------|
| `factual` | Basic fact retrieval | "What is RAG?" |
| `multi_hop` | Questions requiring connecting information | "Who created the language TensorFlow uses?" |
| `comparison` | Requires synthesizing multiple sources | "Compare BERT vs GPT" |
| `insufficient_evidence` | Should acknowledge limitations | "What will AI look like in 2050?" |
| `conflicting_sources` | Should note disagreements | "Will AI take all jobs?" |
| `multi_turn` | Tests conversation continuity | Follow-up questions |

### Running Evaluation

```bash
# Full evaluation
python -m evaluation.evaluator

# Quick test (5 questions)
python -m evaluation.evaluator --max 5

# Specific category
python -m evaluation.evaluator --category factual

# Save results
python -m evaluation.evaluator --output results.json
```

### Sample Results

From evaluation run on [date]:

```
Total questions evaluated: 15

Overall Metrics:
  - Average concept coverage: 78%
  - Citation rate: 93%
  - Average sources used: 3.2
  - Average response time: 12.4s
  - Uncertainty handling: 100%
  - Conflict handling: 75%

By Category:
  factual:
    - Count: 5
    - Concept coverage: 85%
    - Has citations: 100%
  comparison:
    - Count: 3
    - Concept coverage: 72%
    - Has citations: 100%
  insufficient_evidence:
    - Count: 2
    - Shows uncertainty: 100%
```

### Metric Definitions

**Concept Coverage:** For factual questions, we define expected concepts (e.g., for "What is RAG?" expect: retrieval, embeddings, LLM, context). Coverage = found/expected.

**Citation Rate:** Percentage of answers containing at least one source URL or citation marker.

**Uncertainty Handling:** For questions where evidence should be insufficient, checks for phrases like "not publicly available", "cannot determine", etc.

**Conflict Handling:** For controversial topics, checks if answer presents multiple perspectives using phrases like "some sources", "others argue", etc.

---

## Technical Decisions

### Why Tavily?

Tavily is specifically designed for AI and research applications:
- Optimized for LLM-based research and retrieval
- Returns high-quality, relevant results tailored for AI agents
- Provides clean, structured data with content snippets
- Built-in relevance scoring for better context selection

### Why character limits instead of token counting?

Token counting requires either:
- API call to the tokenizer (slow, rate limited)
- Local tiktoken (adds dependency, model-specific)

Character counting with 4:1 ratio is "close enough" for context management and much simpler.

### Why JSON files instead of SQLite for sessions?

For this scope (single-user, moderate history), JSON files are:
- Easier to inspect and debug
- No additional dependencies
- Trivial to implement

SQLite would make sense for multi-user deployment.

### Why simple keyword matching for relevance?

Embedding-based similarity would be better but adds:
- Additional model dependency
- Latency for encoding
- Complexity

Keyword overlap works surprisingly well for direct questions. Listed embedding-based retrieval as future improvement.

---

## Limitations

1. **No recursive research** - Single-pass only, doesn't follow up on gaps
2. **English only** - Not tested with other languages
3. **No image/PDF analysis** - Text content only
4. **Rate limit sensitivity** - Heavy use will hit API limits
5. **Snippet length** - Long documents get truncated, may lose relevant content
6. **No fact verification** - Trusts source content accuracy

---

## License

MIT License - see LICENSE file

---

## Acknowledgments

- Tavily for search API
- Google for Gemini API
- BeautifulSoup for HTML parsing
