# Autonomous Lead Enrichment Agent

An autonomous, local AI-driven intelligence pipeline that accepts company domains, crawls public web surfaces using Playwright, filters high-signal corporate pages, extracts clean text using BeautifulSoup, deterministically discovers verified contact points (emails and LinkedIn profiles), optimizes context, and extracts structured, validated company intelligence using Ollama and Gemma 3 (`gemma3:latest`).

---

## Overview

Modern B2B sales development and lead qualification require accurate, real-time company intelligence (overview, ICP/target audience, verified contacts, and leadership teams). Relying on manual web browsing is slow and inconsistent, while sending raw webpage HTML to LLMs wastes context tokens, causes hallucination, and exposes pipelines to high latency and vendor lock-in.

This project delivers a resilient, fully local, and production-ready enrichment pipeline. It extracts real-time signals directly from public company websites, applies deterministic regex filters for contact points, feeds optimized context into a locally hosted **Gemma 3** model, and enforces strict schema compliance through **Pydantic**.

---

## Problem Statement

When building lead enrichment systems with Large Language Models, engineers face four key challenges:
1. **JavaScript-Rendered SPAs:** Modern tech websites rely on client-side rendering (React, Vue, Next.js). Simple HTTP scrapers fail to render dynamic content.
2. **Context Window Waste & Noise:** Webpages contain thousands of lines of navigation menus, footer links, tracking scripts, and styling tags. Feeding raw HTML into LLMs dilutes prompt relevance and inflates token usage.
3. **Hallucination Risk:** LLMs are prone to fabricating executive names, guessing emails, or constructing plausible-looking LinkedIn profile URLs when prompted without strict constraints.
4. **Pipeline Fragility:** Crawl timeouts, 404s, bot protections, or malformed LLM outputs can crash an entire batch job.

The Autonomous Lead Enrichment Agent solves these issues with headless browser rendering, BeautifulSoup sanitization, deterministic regex evidence collection, context budgeting, and fallback error handling.

---

## Features

- **Headless Chromium Automation:** Uses Playwright to render JavaScript-heavy landing pages and handle dynamic DOM hydration.
- **Heuristic Page Discovery:** Intelligently discovers and prioritizes high-value corporate pages (`about`, `company`, `team`, `leadership`, `contact`, `pricing`, `product`) while filtering noise (`login`, `signup`, `privacy`, `terms`, `docs`, `blog`, `careers`).
- **Clean HTML Stripping:** Decomposes non-content tags (`script`, `style`, `noscript`, `svg`, `nav`, `footer`, `iframe`) via BeautifulSoup to supply pure plain text to the LLM.
- **Deterministic Contact Extraction:** Extracts real public emails and LinkedIn URLs directly from DOM links and text using strict regex filters, preventing the LLM from inventing contacts.
- **Predictable Context Optimization:** Implements character budgeting per page and total context limits to maximize signal-to-noise and prevent context overflow.
- **Local & Private LLM Extraction:** Exclusively runs **Ollama** with `gemma3:latest`, eliminating API costs, rate limits, and third-party data transmission.
- **Strict Pydantic Validation:** Enforces schema integrity on LLM responses with boundary checks (`0.0 <= confidence_score <= 1.0`) and markdown fence stripping.
- **Fault-Tolerant Architecture:** Resilient error handling ensures that failures on a single domain or page never crash the broader batch process.

---

## Architecture

```
                       ┌────────────────────────┐
                       │  Input Company Domain  │
                       │ (postman, supabase...) │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │   Playwright Browser   │
                       │   (Chromium Headless)  │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Relevant Page Discovery│
                       │ (About, Team, Pricing) │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │  BeautifulSoup Parser  │
                       │ (Strip nav/footer/svg) │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │ Deterministic Evidence │
                       │ (Regex Emails/LinkedIn)│
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │  Context Optimization  │
                       │ (Page & Total Budgets) │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │   Ollama / Gemma 3     │
                       │   (gemma3:latest)      │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │  Pydantic Validation   │
                       │  (Strict Schema Guard) │
                       └───────────┬────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │   output/output.json   │
                       └────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core application logic and type annotations |
| **Browser Engine** | Playwright (Chromium) | Headless page rendering and dynamic SPA navigation |
| **HTML Parser** | BeautifulSoup4 | Removal of boilerplate tags and text normalization |
| **LLM Runtime** | Ollama | Local model hosting and fast inference |
| **LLM Model** | Gemma 3 (`gemma3:latest`) | Zero-cost, locally hosted intelligence extraction |
| **Data Validation** | Pydantic v2 | Strict schema compliance, type casting, and constraints |

---

## Project Structure

```
softwarebrio-ai-agent/
│
├── app/
│   ├── __init__.py         # Package marker and namespace
│   ├── browser.py          # Playwright Chromium browser lifecycle and fetch routines
│   ├── crawler.py          # Domain crawler, priority heuristic ranking, and link discovery
│   ├── extractor.py        # BeautifulSoup HTML cleaning, sanitization, and contact extraction
│   ├── llm.py              # Ollama Gemma 3 integration, prompt engineering, and parsing
│   ├── main.py             # Pipeline orchestrator, logging setup, and batch execution
│   ├── schemas.py          # Pydantic models (CompanyData, LeadershipMember) and fallback generators
│   └── utils.py            # Helper utilities: URL normalization, regex, context budgeting, JSON repair
│
├── output/
│   └── output.json         # Enriched structured intelligence JSON output
│
├── .gitignore              # Ignored files (venv, pycache, .env)
├── README.md               # Architecture and operational documentation
└── requirements.txt        # Core project dependencies
```

---

## Prerequisites

Before running the application, ensure the following are installed on your host system:
1. **Python 3.10+**
2. **Playwright** & **Chromium browser binaries**
3. **Ollama** installed and running locally
4. **Gemma 3 model** pulled in Ollama

---

## Ollama Setup

Verify that the Ollama daemon is running and that `gemma3:latest` is available:

1. Start Ollama (if not already running as a background service):
   ```bash
   ollama serve
   ```

2. Verify available models:
   ```bash
   ollama list
   ```

3. If `gemma3:latest` is not present, pull it:
   ```bash
   ollama pull gemma3:latest
   ```

4. Test that the model responds:
   ```bash
   ollama run gemma3:latest "Hello"
   ```

---

## Running the Project

Ensure your virtual environment is active and run the pipeline:

```bash
python -m app.main
```

### Demonstration Logs
During execution, clean logs indicate progress across all phases:
```
==================================================
Starting Autonomous Lead Enrichment Agent Pipeline
Target Companies: postman.com, supabase.com, vapi.ai
==================================================
--------------------------------------------------
[INFO] Processing postman.com
[INFO] Homepage fetched (200)
[INFO] Relevant pages discovered: 5 (Total internal: 28)
[INFO] Content cleaned across 6 pages
[INFO] Emails discovered: 1
[INFO] LinkedIn URLs discovered: 1
[INFO] Sending context to Gemma 3 (gemma3:latest)
[INFO] LLM response received
[INFO] Pydantic validation successful
...
==================================================
[INFO] Pipeline complete! Output successfully written to: ...\output\output.json
[INFO] Processed 3 companies.
==================================================
```

---

## Output Format

Results are saved to `output/output.json` as a formatted JSON array conforming to the `CompanyData` Pydantic schema:

```json
[
  {
    "domain": "postman.com",
    "company_overview": "Postman is an API platform for building, testing, and managing APIs across the complete development lifecycle. It simplifies API collaboration and streamlines workflow automation for developers and enterprise teams.",
    "target_audience": "Software developers, API engineers, product managers, and enterprise engineering teams building or integrating REST, GraphQL, or gRPC APIs.",
    "contact_points": [
      "help@postman.com"
    ],
    "leadership": [
      {
        "name": "Abhinav Asthana",
        "role": "CEO and Co-founder",
        "linkedin_url": "https://www.linkedin.com/company/postman-platform"
      }
    ],
    "confidence_score": 0.95
  }
]
```

### Schema Attributes:
- **`domain`** *(str)*: Target company domain.
- **`company_overview`** *(str)*: Concise two-sentence explanation of what the company does.
- **`target_audience`** *(str)*: Identified Ideal Customer Profile (ICP) and intended users.
- **`contact_points`** *(list[str])*: Public/generic contact emails discovered on the website.
- **`leadership`** *(list[LeadershipMember])*: Array of executives with `name`, `role`, and optional verified `linkedin_url`.
- **`confidence_score`** *(float)*: Strictly bounded between `0.0` and `1.0`.

---

## Context & Token Optimization Strategy

To ensure deterministic performance, low latency, and zero token-budget truncation errors in Gemma 3, the agent applies a 3-tier optimization strategy:

1. **Selective Crawling & Noise Exclusion:**
   - Filters out non-informative paths (`/login`, `/signup`, `/privacy`, `/terms`, `/blog`, `/careers`, `/docs`) before fetching.
   - Restricts crawling to a maximum of 6 pages per domain (`max_pages=6`), starting with the homepage and ranking candidate pages by keywords (`leadership`, `team`, `about`, `company`, `contact`, `pricing`, `product`).

2. **DOM Noise Decomposition:**
   - Removes all markup tags (`<script>`, `<style>`, `<nav>`, `<footer>`, `<svg>`, `<iframe>`, `<form>`) using BeautifulSoup.
   - Extracts plain visible text while preserving logical paragraph and heading line breaks.

3. **Character Budgeting & Deduplication:**
   - **Per-Page Ceiling:** Truncates any individual page content to 3,500 characters.
   - **Total Context Budget:** Enforces an aggregate ceiling of 12,000 characters across all combined pages per company.
   - **Chunk Deduplication:** Tracks structural fingerprints across pages to avoid repeating shared legal or banner snippets.

---

## Error Handling & Resiliency

- **Network & Browser Failures:** Playwright page navigations are wrapped with timeouts (default 25s) and cleanup routines. If a page or domain times out or triggers bot-detection, the crawler logs `[WARNING] Could not fetch...` and continues to the next candidate page.
- **Domain-Level Isolation:** An exception on one domain is caught and isolated in `app/main.py`. The agent generates a safe fallback record with `confidence_score: 0.0` and proceeds to the remaining companies.
- **Markdown Fence & JSON Sanitization:** The LLM's raw text is stripped of markdown code blocks (` ```json `), and regex bracket-matching locates the outermost JSON object before parsing.
- **Strict Pydantic Validation:** If the LLM generates an invalid schema or violates constraints (e.g. `confidence_score` out of bounds), validation triggers the safe fallback generator (`create_fallback_company_data`) rather than crashing the pipeline.

---

## Limitations

- **Anti-Scraping / Cloudflare Captchas:** Extremely aggressive Cloudflare challenge screens or hard IP bans may block headless Chromium requests.
- **Deeply Buried Leadership:** Companies that list leadership teams exclusively on external investor portals or separate microsites may return empty leadership arrays since the crawler strictly enforces same-domain boundaries.
- **Local Model Hardware Requirements:** Running `gemma3:latest` requires sufficient local RAM / VRAM (typically 8GB+).

---

## Future Enhancements

- **Sitemap.xml Ingestion:** Supplement link discovery with direct `sitemap.xml` parsing for faster internal page routing.
- **Configurable Proxy Rotation:** Integrate rotating residential proxies to bypass bot-detection on enterprise sites.
- **Asynchronous Crawl Worker Pool:** Expand Playwright crawling to concurrent browser pages for large-scale batches.
