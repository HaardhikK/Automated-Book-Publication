# 📚 Automated Book Publication

An end-to-end modular framework that automates the lifecycle of publishing rewritten book chapters—from scraping and rewriting to human editing, versioning, and intelligent search.

---

## 🧠 Core Modules & Design Highlights

### 🔹 Part 1: **Wikisource Scraper**

**Script:** `part1_scraper.py`
**Tool:** Playwright (headless browser)

**Key Features:**

* Extracts clean content from Wikisource chapters by targeting precise HTML selectors.
* Captures a **full-page screenshot** of each chapter for archival and visual inspection.
* Implements **fallback extraction**:

  * If paragraph selectors fail, it pulls all visible text.
  * If screenshot fails, logs the issue and continues with content processing.
* Logs output path, content length, and errors for traceability.

### 🔹 Part 2: **AI Writer**

**Script:** `part2_ai_writer.py`
**Model:** `Llama3-70B` via Groq API

**Key Features:**

* Rewrites the scraped chapter into a more engaging, vivid version.
* Maintains **core story, character names, and narrative structure**.
* Automatically filters AI hallucinations or unwanted commentary.
* Logs and stores:

  * Input/output text length
  * Token usage
  * Prompt used
  * Compression ratio
* Supports **robust error handling**:

  * Detects and alerts if the input file is missing or empty.
  * Prevents continuation if no valid output is generated.

### 🔹 Part 3: **AI Reviewer**

**Script:** `part3_ai_reviewer.py`
**Model:** `Deepseek-R1` via Groq API

**Key Features:**

* Performs a second-pass professional polish on AI-written content.
* Focuses on **grammar, tone, flow, punctuation, and structure** without altering meaning.
* Strips out any unwanted model commentary or internal tags (`<think>` etc.).
* Uses a lower temperature for **maximum consistency and editing reliability**.
* Stores a rich metadata file with all review details, version, length changes, and quality metrics.

### 🔹 Part 4: **Human-in-the-Loop Editing UI**

**Script:** `part4_human_interface.py`
**Tool:** Streamlit

**Key Features:**

* GUI interface to:

  * Compare different chapter versions side-by-side
  * See visual diffs (with color-coded highlights for added/removed/modified text)
  * Edit chapters manually and save new versions
* Tracks **version lineage**: saves source version, editor name, and timestamp.
* Provides a fallback diff view (text-based) in case rich HTML rendering fails.
* Enforces version control: avoids accidental overwrite of earlier edits.

### 🔹 Part 5: **Agentic Flow Orchestration**

**Script:** `part5_agent_flow_orchestration.py`

**Key Features:**

* Sequentially runs the full pipeline (Scrape → AI Write → AI Review).
* Maintains a unique **flow UUID** for every batch.
* Logs every step with:

  * Status (started, failed, success)
  * Input/output files
  * Metadata including tokens used, content length, and model used
* **Gracefully handles failures**:

  * Skips a chapter’s next stage if the current one fails.
  * Logs detailed errors and continues with other chapters.

### 🔹 Part 6: **Versioning + RL-Driven Semantic Search**

**Script:** `part6_faiss_versioning.py`
**Components:** FAISS + SentenceTransformers + Custom RL Layer

**Key Features:**

* **Embeds all chapter versions** (`ai`, `reviewed`, `human`, `final`) into a FAISS vector index.
* Supports **semantic retrieval** using queries like “version with best flow” or “human edit with most vivid descriptions”.
* Uses **metadata parsing** (from filenames) to infer chapter, version type, and editor.
* Includes a **Reinforcement Learning (RL) feedback mechanism**:

  * Boosts document ranking based on user feedback over time.
  * Merges feedback from similar queries for generalization.
  * Keeps score history and learns dynamically.

**Fallbacks:**

* Skips and logs corrupt/empty text files.
* Avoids re-indexing already stored versions.
* Resilient against malformed metadata.

---


