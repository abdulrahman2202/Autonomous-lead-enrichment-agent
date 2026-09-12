"""Reusable utility functions for the Autonomous Lead Enrichment Agent.

Provides URL normalization, link relevance filtering, deterministic email and
LinkedIn URL extraction, context token optimization, and JSON response cleanup.
"""

import json
import re
from urllib.parse import urljoin, urlparse


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
    # Strip port if present
    if ":" in netloc:
        netloc = netloc.split(":")[0]
    # Strip leading www. if present for domain matching comparison
    return netloc


def build_base_url(domain: str) -> str:
    """Builds a standardized HTTPS base URL from a domain string.

    Args:
        domain: Domain name (e.g., 'postman.com').

    Returns:
        Standard HTTPS base URL (e.g., 'https://postman.com').
    """
    clean_domain = normalize_domain(domain)
    return f"https://{clean_domain}"


def is_same_domain(url: str, base_domain: str) -> bool:
    """Verifies whether a URL belongs to the target domain or its subdomains.

    Args:
        url: Absolute URL to check.
        base_domain: Target company domain.

    Returns:
        True if the host matches or is a direct subdomain of base_domain.
    """
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

    # Check file extensions
    if any(parsed.path.endswith(ext) for ext in EXCLUDED_EXTENSIONS):
        return True

    # Check path segments and query for excluded keywords
    path_and_query = f"{parsed.path}?{parsed.query}"
    return any(keyword in path_and_query for keyword in EXCLUDED_KEYWORDS)


def normalize_internal_url(href: str, current_page_url: str, base_domain: str) -> str | None:
    """Normalizes and filters candidate internal links.

    Args:
        href: Extracted href attribute.
        current_page_url: The URL of the page where href was discovered.
        base_domain: The target company domain.

    Returns:
        Canonical absolute HTTPS URL if internal and eligible, else None.
    """
    if not href:
        return None

    cleaned_href = href.strip()
    # Ignore non-http links
    if cleaned_href.startswith(("mailto:", "tel:", "javascript:", "#", "data:")):
        return None

    absolute_url = urljoin(current_page_url, cleaned_href)
    parsed = urlparse(absolute_url)

    if parsed.scheme not in ("http", "https"):
        return None

    # Enforce HTTPS scheme and strip fragments
    canonical_url = f"https://{parsed.netloc}{parsed.path}"
    if parsed.query:
        canonical_url = f"{canonical_url}?{parsed.query}"

    # Remove trailing slash for consistency (unless root)
    if canonical_url.endswith("/") and parsed.path != "/":
        canonical_url = canonical_url[:-1]

    # Enforce domain boundary
    if not is_same_domain(canonical_url, base_domain):
        return None

    # Exclude non-informative paths
    if is_excluded_url(canonical_url):
        return None

    return canonical_url


def extract_emails_from_text(text: str) -> list[str]:
    """Deterministically extracts valid public email addresses using regex.

    Excludes false positives such as image files or script artifacts.

    Args:
        text: Raw or cleaned webpage text.

    Returns:
        Sorted, deduplicated list of lowercase email addresses.
    """
    if not text:
        return []

    matches = EMAIL_REGEX.findall(text)
    valid_emails = set()

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

    for match in matches:
        cleaned = match.strip().lower()
        # Filter out asset false-positives (e.g. image@2x.png)
        if any(cleaned.endswith(ext) for ext in invalid_extensions):
            continue
        # Filter out obvious mock / template emails
        if cleaned.endswith((".example.com", ".test.com", "@domain.com", "@email.com")):
            continue
        valid_emails.add(cleaned)

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
        # Enforce HTTPS
        if cleaned.startswith("http://"):
            cleaned = "https://" + cleaned[len("http://") :]
        # Remove trailing slash
        if cleaned.endswith("/"):
            cleaned = cleaned[:-1]
        valid_urls.add(cleaned)

    return sorted(valid_urls)


def optimize_context(
    pages_data: list[dict],
    max_chars_per_page: int = 3500,
    max_total_chars: int = 12000,
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

        # Truncate text per page
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


def clean_and_parse_json(raw_text: str) -> dict | None:
    """Extracts and parses JSON from raw LLM text.

    Handles Markdown code fences (```json ... ```), surrounding prose,
    and outer bracket boundary discovery.

    Args:
        raw_text: The raw text response received from the LLM.

    Returns:
        Parsed dictionary if successful, None otherwise.
    """
    if not raw_text or not raw_text.strip():
        return None

    cleaned = raw_text.strip()

    # Step 1: Strip markdown code fences if present
    if "```" in cleaned:
        # Match ```json ... ``` or ``` ... ```
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()

    # Step 2: Try direct JSON parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Step 3: Find outermost balanced curly brackets
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate_json = cleaned[start_idx : end_idx + 1]
        try:
            parsed = json.loads(candidate_json)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return None
