"""HTML cleaning and structured webpage content extraction using BeautifulSoup.

Removes boilerplate, styling, scripts, and navigation markup to produce clean,
readable plain text, while discovering internal links, emails, and LinkedIn profiles.
"""

import re
from bs4 import BeautifulSoup, Comment
from app.utils import (
    extract_emails_from_text,
    extract_linkedin_urls_from_text,
    normalize_internal_url,
)

TAGS_TO_REMOVE = [
    "script",
    "style",
    "noscript",
    "svg",
    "nav",
    "footer",
    "iframe",
    "form",
    "button",
    "select",
    "option",
]


def extract_page_title(soup: BeautifulSoup) -> str:
    """Extracts a sanitized page title."""
    if soup.title and soup.title.string:
        return " ".join(soup.title.string.split())
    h1 = soup.find("h1")
    if h1:
        return " ".join(h1.get_text().split())
    return "Untitled Page"


def clean_html_to_text(html_content: str) -> str:
    """Cleans raw HTML and extracts readable plain text suitable for LLM context.

    Strips scripts, stylesheets, navigation bars, footers, and SVGs.
    Preserves heading and paragraph hierarchy while eliminating redundant whitespace.

    Args:
        html_content: Raw HTML string.

    Returns:
        Cleaned, high-signal plain text.
    """
    if not html_content or not html_content.strip():
        return ""

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove comments
    for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove unwanted tags
    for tag_name in TAGS_TO_REMOVE:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Insert spacing around block-level elements before text extraction
    block_tags = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "article", "section", "div"]
    for tag in soup.find_all(block_tags):
        tag.insert_before(" ")
        tag.insert_after("\n")

    text = soup.get_text(separator=" ")

    # Collapse excessive blank lines and horizontal whitespace
    lines = [line.strip() for line in text.splitlines()]
    clean_lines = [line for line in lines if line]
    formatted_text = "\n".join(clean_lines)

    # Replace runs of 3+ newlines with double newlines
    return re.sub(r"\n{3,}", "\n\n", formatted_text).strip()


def extract_page_data(html_content: str, current_url: str, base_domain: str) -> dict:
    """Extracts clean text, internal links, public emails, and LinkedIn URLs from HTML.

    Args:
        html_content: Raw HTML response.
        current_url: The canonical URL of the current page.
        base_domain: Company base domain.

    Returns:
        Dictionary containing title, cleaned_text, internal_links, emails, and linkedin_urls.
    """
    if not html_content:
        return {
            "url": current_url,
            "title": "",
            "cleaned_text": "",
            "internal_links": [],
            "emails": [],
            "linkedin_urls": [],
        }

    soup = BeautifulSoup(html_content, "html.parser")
    title = extract_page_title(soup)

    # 1. Deterministic Link & Contact Discovery from <a> tags
    internal_links_set: set[str] = set()
    emails_set: set[str] = set()
    linkedin_set: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()

        # Check mailto: links
        if href.lower().startswith("mailto:"):
            email_part = href[7:].split("?")[0].strip()
            discovered = extract_emails_from_text(email_part)
            emails_set.update(discovered)
            continue

        # Check LinkedIn URLs
        if "linkedin.com" in href.lower():
            discovered_li = extract_linkedin_urls_from_text(href)
            linkedin_set.update(discovered_li)

        # Check for eligible internal link
        normalized_url = normalize_internal_url(href, current_url, base_domain)
        if normalized_url and normalized_url != current_url:
            internal_links_set.add(normalized_url)

    # 2. Clean HTML to human-readable plain text
    cleaned_text = clean_html_to_text(html_content)

    # 3. Deterministic Extraction from page text and raw HTML
    text_emails = extract_emails_from_text(cleaned_text)
    raw_emails = extract_emails_from_text(html_content)
    emails_set.update(text_emails)
    emails_set.update(raw_emails)

    text_linkedin = extract_linkedin_urls_from_text(html_content)
    linkedin_set.update(text_linkedin)

    return {
        "url": current_url,
        "title": title,
        "cleaned_text": cleaned_text,
        "internal_links": sorted(internal_links_set),
        "emails": sorted(emails_set),
        "linkedin_urls": sorted(linkedin_set),
    }
