"""Reusable utility functions for the Autonomous Lead Enrichment Agent.

Provides URL normalization, link relevance filtering, deterministic email and
LinkedIn URL extraction, context token optimization, and robust JSON response parsing.
"""

import html
import json
import logging
import re
from urllib.parse import urljoin, urlparse

logger = logging.getLogger("lead_enrichment")

PRIORITY_KEYWORDS = [
    "about",
    "company",
    "team",
    "leadership",
    "contact",
    "pricing",
    "product",
]

EXCLUDED_KEYWORDS = [
    "login",
    "signin",
    "signup",
    "register",
    "privacy",
    "terms",
    "tos",
    "policy",
    "blog",
    "news",
    "article",
    "careers",
    "jobs",
    "docs",
    "documentation",
    "status",
    "download",
    "help",
    "support",
    "faq",
    "feed",
]

EXCLUDED_EXTENSIONS = (
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".css",
    ".js",
    ".xml",
    ".rss",
)

EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

LINKEDIN_REGEX = re.compile(
    r"https?://(?:[a-z]{2,3}\.)?linkedin\.com/(?:in|company)/[a-zA-Z0-9_\-%]+/?",
    re.IGNORECASE,
)


def normalize_domain(domain_or_url: str) -> str:
    """Extracts a clean, normalized domain name.

    Args:
        domain_or_url: Raw domain or URL (e.g., 'https://www.postman.com/path').

    Returns:
        Clean domain string in lowercase without scheme, path, or port.
    """
    raw = domain_or_url.strip().lower()
    if not raw.startswith(("http://", "https://")):
        raw = "https://" + raw

    parsed = urlparse(raw)
    netloc = parsed.netloc or parsed.path
    if ":" in netloc:
        netloc = netloc.split(":")[0]
    return netloc


def build_base_url(domain: str) -> str:
    """Builds a standardized HTTPS base URL from a domain string."""
    clean_domain = normalize_domain(domain)
    return f"https://{clean_domain}"


def is_same_domain(url: str, base_domain: str) -> bool:
    """Verifies whether a URL belongs to the target domain or its subdomains."""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    clean_base = base_domain.lower().replace("www.", "")

    if ":" in host:
        host = host.split(":")[0]

    return host == clean_base or host == f"www.{clean_base}" or host.endswith(f".{clean_base}")


def is_priority_url(url: str) -> bool:
    """Checks if a URL contains high-priority keywords for company intelligence."""
    lowered = url.lower()
    return any(keyword in lowered for keyword in PRIORITY_KEYWORDS)


def is_excluded_url(url: str) -> bool:
    """Checks if a URL matches common non-informative or redundant patterns."""
    lowered = url.lower()
    parsed = urlparse(lowered)

    if any(parsed.path.endswith(ext) for ext in EXCLUDED_EXTENSIONS):
        return True

    path_and_query = f"{parsed.path}?{parsed.query}"
    return any(keyword in path_and_query for keyword in EXCLUDED_KEYWORDS)


def normalize_internal_url(href: str, current_page_url: str, base_domain: str) -> str | None:
    """Normalizes and filters candidate internal links."""
    if not href:
        return None

    cleaned_href = href.strip()
    if cleaned_href.startswith(("mailto:", "tel:", "javascript:", "#", "data:")):
        return None

    absolute_url = urljoin(current_page_url, cleaned_href)
    parsed = urlparse(absolute_url)

    if parsed.scheme not in ("http", "https"):
        return None

    canonical_url = f"https://{parsed.netloc}{parsed.path}"
    if parsed.query:
        canonical_url = f"{canonical_url}?{parsed.query}"

    if canonical_url.endswith("/") and parsed.path != "/":
        canonical_url = canonical_url[:-1]

    if not is_same_domain(canonical_url, base_domain):
        return None

    if is_excluded_url(canonical_url):
        return None

    return canonical_url


def clean_email_candidate(raw_candidate: str) -> str | None:
    """Decodes, cleans, and validates a potential email string.

    Handles HTML entities (&gt;, &lt;), unicode escapes (\\u003e, u003e),
    stray prefixes, and enforces strict local-part@domain.tld formatting.

    Args:
        raw_candidate: Raw candidate string matching an email pattern.

    Returns:
        Clean, lowercase, validated email address or None if invalid.
    """
    if not raw_candidate:
        return None

    # Step 1: Decode HTML entities (&gt;, &lt;, &#x3e;, etc.)
    candidate = html.unescape(raw_candidate.strip())

    # Step 2: Decode raw unicode escape strings like \u003e or \u003c
    candidate = re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda m: chr(int(m.group(1), 16)),
        candidate,
    )

    # Step 3: Strip URL encodings (%3E, %3C, %20)
    candidate = (
        candidate.replace("%3e", ">")
        .replace("%3E", ">")
        .replace("%3c", "<")
        .replace("%3C", "<")
    )

    # Step 4: Strip leading unwanted artifacts (e.g. u003e, u003c, &gt;, mailto:, brackets)
    candidate = re.sub(
        r"^(?:u003[ce]|u002[26]|gt|lt|mailto|[<>\(\)\[\]\"'#;:+=%/\\])+",
        "",
        candidate,
        flags=re.IGNORECASE,
    )

    # Step 5: Strip trailing brackets and punctuation
    candidate = re.sub(r"[<>\(\)\[\]\"'#;:+=%/\\]+$", "", candidate)

    email = candidate.strip().lower()

    # Step 6: Strict validation: local-part@domain.tld
    strict_pattern = r"^[a-z0-9][a-z0-9._%+-]*@[a-z0-9][a-z0-9.-]*\.[a-z]{2,}$"
    if not re.match(strict_pattern, email):
        return None

    # Step 7: Filter out asset and script false-positives
    invalid_extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".webp",
        ".js",
        ".css",
        ".ts",
    )
    if any(email.endswith(ext) for ext in invalid_extensions):
        return None

    # Step 8: Filter out dummy / mock templates
    if email.endswith((".example.com", ".test.com", "@domain.com", "@email.com")):
        return None

    return email


def extract_emails_from_text(text: str) -> list[str]:
    """Deterministically extracts valid public email addresses.

    Normalizes HTML/Unicode representations and strips unwanted prefixes.

    Args:
        text: Raw HTML or cleaned webpage text.

    Returns:
        Sorted, deduplicated list of lowercase email addresses.
    """
    if not text:
        return []

    # First decode HTML entities to prevent mangled boundaries
    decoded_text = html.unescape(text)
    matches = EMAIL_REGEX.findall(decoded_text)
    valid_emails = set()

    for match in matches:
        cleaned_email = clean_email_candidate(match)
        if cleaned_email:
            valid_emails.add(cleaned_email)

    return sorted(valid_emails)


def extract_linkedin_urls_from_text(text: str) -> list[str]:
    """Deterministically extracts legitimate LinkedIn URLs using regex.

    Args:
        text: Raw HTML or webpage text.

    Returns:
        Sorted, deduplicated list of canonical LinkedIn URLs.
    """
    if not text:
        return []

    matches = LINKEDIN_REGEX.findall(text)
    valid_urls = set()

    for match in matches:
        cleaned = match.strip()
        if cleaned.startswith("http://"):
            cleaned = "https://" + cleaned[len("http://") :]
        if cleaned.endswith("/"):
            cleaned = cleaned[:-1]
        valid_urls.add(cleaned)

    return sorted(valid_urls)


def optimize_context(
    pages_data: list[dict],
    max_chars_per_page: int = 2500,
    max_total_chars: int = 8000,
) -> str:
    """Optimizes scraped website content into a concise, high-signal prompt context.

    Applies per-page length limits, deduplication, and a total character ceiling
    to ensure predictable token usage within Gemma 3.

    Args:
        pages_data: List of dicts with keys 'url', 'title', and 'cleaned_text'.
        max_chars_per_page: Maximum characters retained from a single page.
        max_total_chars: Maximum characters across all combined pages.

    Returns:
        Structured, consolidated text context suitable for the LLM.
    """
    chunks: list[str] = []
    current_total_len = 0
    seen_texts: set[str] = set()

    for page in pages_data:
        url = page.get("url", "")
        title = page.get("title", "")
        raw_text = page.get("cleaned_text", "").strip()

        if not raw_text:
            continue

        truncated_page_text = raw_text[:max_chars_per_page].strip()
        text_fingerprint = truncated_page_text[:200]

        if text_fingerprint in seen_texts:
            continue
        seen_texts.add(text_fingerprint)

        header = f"=== SOURCE PAGE: {url} (Title: {title}) ==="
        page_block = f"{header}\n{truncated_page_text}\n"

        if current_total_len + len(page_block) > max_total_chars:
            remaining_budget = max_total_chars - current_total_len
            if remaining_budget > 200:
                header_len = len(header) + 2
                allowed_text_len = max(0, remaining_budget - header_len)
                trimmed_text = truncated_page_text[:allowed_text_len].rsplit("\n", 1)[0]
                chunks.append(f"{header}\n{trimmed_text}...\n")
            break

        chunks.append(page_block)
        current_total_len += len(page_block)

    return "\n".join(chunks)


def extract_and_parse_json(raw_text: str) -> dict | None:
    """Robustly extracts and parses a JSON dictionary from raw LLM text.

    Uses Python's standard json.JSONDecoder(strict=False).raw_decode() to support:
    - Plain JSON objects
    - JSON surrounded by leading/trailing whitespace or conversational prose
    - Markdown code fences (```json ... ```) with or without closing fences
    - Nested JSON objects and arrays
    - Escaped characters and unescaped control characters/newlines inside strings
    - Embedded curly braces inside string values without breaking

    If all parsing attempts fail, logs a diagnostic warning with the error message,
    position, and snippet around the failure.

    Args:
        raw_text: The raw text response received from the LLM.

    Returns:
        Parsed dictionary if a valid JSON object is found, None otherwise.
    """
    if not raw_text or not isinstance(raw_text, str):
        return None

    cleaned = raw_text.strip()
    if not cleaned:
        return None

    # Step 1: Direct parse if complete response is valid JSON
    try:
        parsed = json.loads(cleaned, strict=False)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Step 2: Strip markdown code fences if present (supports unclosed fences as well)
    stripped_fence = cleaned
    if "```" in cleaned:
        fence_match = re.search(
            r"```(?:json)?\s*([\s\S]*?)(?:```|$)", cleaned, re.IGNORECASE
        )
        if fence_match and fence_match.group(1).strip():
            stripped_fence = fence_match.group(1).strip()
            try:
                parsed = json.loads(stripped_fence, strict=False)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

    # Step 3: Robust scanning via json.JSONDecoder(strict=False).raw_decode
    decoder = json.JSONDecoder(strict=False)
    candidate_texts = []
    if stripped_fence != cleaned:
        candidate_texts.append(stripped_fence)
    candidate_texts.append(cleaned)
    if raw_text not in candidate_texts:
        candidate_texts.append(raw_text)

    last_error: json.JSONDecodeError | None = None
    last_error_text: str = ""

    for candidate_str in candidate_texts:
        idx = 0
        str_len = len(candidate_str)
        while idx < str_len:
            brace_pos = candidate_str.find("{", idx)
            if brace_pos == -1:
                break
            try:
                obj, end_pos = decoder.raw_decode(candidate_str, brace_pos)
                if isinstance(obj, dict):
                    return obj
                idx = brace_pos + 1
            except json.JSONDecodeError as jde:
                if last_error is None or jde.pos > last_error.pos:
                    last_error = jde
                    last_error_text = candidate_str
                idx = brace_pos + 1
            except Exception:
                idx = brace_pos + 1

    # Step 4: Safe repair attempts for common LLM JSON defects
    repaired = stripped_fence

    # 4a. Remove trailing commas before } or ]
    repaired = re.sub(r",\s*([\]}])", r"\1", repaired)

    # 4b. Fix invalid escaped single quotes (\' is invalid in standard JSON)
    repaired = repaired.replace(r"\'", "'")

    # 4c. Fix truncated / unclosed JSON: balance unclosed strings and braces
    in_str = False
    escape = False
    open_brackets: list[str] = []
    for ch in repaired:
        if ch == "\\" and in_str:
            escape = not escape
            continue
        if ch == '"' and not escape:
            in_str = not in_str
        elif not in_str:
            if ch in "{[":
                open_brackets.append(ch)
            elif ch in "}]":
                if open_brackets:
                    matching = "{" if ch == "}" else "["
                    if open_brackets[-1] == matching:
                        open_brackets.pop()
        escape = False

    if in_str:
        repaired += '"'
    while open_brackets:
        needed = open_brackets.pop()
        repaired += "}" if needed == "{" else "]"

    # Test repaired candidate
    try:
        parsed = json.loads(repaired, strict=False)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    idx = 0
    str_len = len(repaired)
    while idx < str_len:
        brace_pos = repaired.find("{", idx)
        if brace_pos == -1:
            break
        try:
            obj, end_pos = decoder.raw_decode(repaired, brace_pos)
            if isinstance(obj, dict):
                return obj
            idx = brace_pos + 1
        except json.JSONDecodeError as jde:
            if last_error is None or jde.pos > last_error.pos:
                last_error = jde
                last_error_text = repaired
            idx = brace_pos + 1
        except Exception:
            idx = brace_pos + 1

    # Step 5: If all attempts fail, log diagnostic error information
    if last_error is not None:
        pos = last_error.pos
        src = last_error_text or raw_text
        start = max(0, pos - 40)
        end = min(len(src), pos + 40)
        snippet = src[start:end].replace("\n", "\\n").replace("\r", "\\r")
        logger.warning(f"[WARNING] JSON parse failed at position {pos}: {last_error.msg}")
        logger.warning(f"[WARNING] Context around failure: ...{snippet}...")
    else:
        logger.warning("[WARNING] JSON parse failed: No valid JSON object could be located.")

    return None


# Backward-compatibility alias
clean_and_parse_json = extract_and_parse_json
