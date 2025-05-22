### Implemented by Ugnius Motiejunas for bachelor thesis

## 📦 LLM-based Web Scraping tool

This tool is designed for automated extraction of structured data (e.g., product names and prices) from e-commerce websites using Large Language Models (LLMs) and the `Crawl4AI` library. It supports asynchronous multiprocessing, enabling the processing of hundreds of pages simultaneously.

---

## 🔧 Features

- Load and read HTML content from a list of URLs.
- Filter irrelevant page elements using `Pruning` or `BM25` content filters.
- Send the processed content to a selected LLM model (e.g., OpenAI GPT or DeepSeek) with a schema and instructions.
- Extract structured JSON output (`name`, `price`, `error`).
- Evaluate the accuracy of the extracted results by comparing them with manually prepared ground-truth data.

---

## 🚀 Getting Started

1. In your `.env` file insert your LLM API keys:

### Using OpenAI:

```env
OPENAI_API_KEY=your_openai_key_here
```

### Using DeepSeek:

```env
DEEPSEEK_API_KEY=your_deepseek_key_here
```

2. Run the main script:

```bash
python crawlnew.py
```

**Arguments:**

- `--batch` – number of parallel processes to run.
- `--use-bm25` – use BM25 content filter instead of Pruning.
- `--query` – prompt that defines what kind of data to extract.

The results will be saved to the JSON file specified in the `--out` argument (default: `combined_results.json`).

---

## 📁 Directory Structure

- `urls.txt` – List of URLs to scrape.
- `expected/` – Folder with manually prepared ground-truth data for each URL.
- `truths.txt` – List that maps each URL to its corresponding `expected/*.json` file.
- `combined_results_*.json` – Model-generated results:
  - `combined_results4.1nano.json` – Results using `gpt-4.1-nano`.
  - `1mini.json` – Results using `gpt-4.1-mini`.
  - `combined_results_deepseek.json` – Results using `deepseek-chat`.
  - `combined_results_iphone_deepseek.json` – iPhone-specific dataset using `deepseek-chat`.

---

## 📊 Evaluation

Results can be evaluated by comparing generated data with ground-truth files in the `expected/` folder. Metrics:

- **TP (True Positives)** – correct matches.
- **FP (False Positives)** – incorrect results.
- **FN (False Negatives)** – missed results.

Metrics calculated:

- **Precision**
- **Recall**
- **F1-score**

---

## 🧠 Use Cases

This tool can be used for:

- Automated price monitoring (competitive intelligence)
- E-commerce data analysis solutions
- Evaluating LLM accuracy in real-world web scraping scenarios
