"""Domain crawler module for the Autonomous Lead Enrichment Agent.

Crawls public company domains using Playwright Chromium, discovers relevant internal
pages, filters high-signal corporate links, extracts clean text, and handles failures
gracefully without crashing the pipeline.
"""

import logging
from playwright.sync_api import Browser
from app.browser import fetch_page_content
from app.extractor import extract_page_data
from app.utils import (
    build_base_url,
    is_priority_url,
    normalize_domain,
)

logger = logging.getLogger("lead_enrichment")

# Priority ranking score for internal URLs
PRIORITY_SCORES = {
    "leadership": 10,
    "team": 9,
    "about": 8,
    "company": 7,
    "contact": 6,
    "pricing": 5,
    "product": 4,
}


def score_url_relevance(url: str) -> int:
    """Calculates a relevance priority score for a discovered internal URL."""
    lowered = url.lower()
    score = 0
    for keyword, weight in PRIORITY_SCORES.items():
        if keyword in lowered:
            score += weight
    return score


def crawl_domain(
    domain: str,
    browser: Browser,
    max_pages: int = 6,
) -> dict:
    """Crawls a company domain, starting from the homepage, to discover relevant pages.

    Prioritizes corporate information pages (about, leadership, team, contact, pricing)
    and aggregates cleaned text, public emails, and verified LinkedIn URLs.

    Args:
        domain: Target company domain (e.g. 'postman.com').
        browser: Active Playwright Chromium Browser instance.
        max_pages: Maximum number of pages to crawl per domain. Defaults to 6.

    Returns:
        Dictionary containing:
            - domain: Normalized domain
            - pages: List of crawled page data dicts
            - combined_cleaned_text: Consolidated plain text from all pages
            - discovered_emails: Deduplicated list of public emails
            - discovered_linkedin_urls: Deduplicated list of LinkedIn URLs
            - error: Error message if domain crawl completely failed, else None
    """
    clean_domain = normalize_domain(domain)
    homepage_url = build_base_url(clean_domain)

    logger.info(f"[INFO] Processing {clean_domain}")

    crawled_pages: list[dict] = []
    visited_urls: set[str] = set()
    candidate_urls: list[str] = []
    all_emails: set[str] = set()
    all_linkedin_urls: set[str] = set()

    # Step 1: Crawl Homepage
    homepage_html, status_code, err = fetch_page_content(browser, homepage_url)

    if err or not homepage_html:
        error_msg = f"Could not fetch homepage for {clean_domain}: {err or 'Empty response'}"
        logger.warning(f"[WARNING] {error_msg}")
        return {
            "domain": clean_domain,
            "pages": [],
            "combined_cleaned_text": "",
            "discovered_emails": [],
            "discovered_linkedin_urls": [],
            "error": error_msg,
        }

    visited_urls.add(homepage_url)
    visited_urls.add(homepage_url + "/")
    logger.info(f"[INFO] Homepage fetched ({status_code or 'OK'})")

    home_data = extract_page_data(homepage_html, homepage_url, clean_domain)
    crawled_pages.append(home_data)
    all_emails.update(home_data["emails"])
    all_linkedin_urls.update(home_data["linkedin_urls"])

    # Step 2: Discover and Prioritize Internal Links
    discovered_links = home_data.get("internal_links", [])
    priority_links = [url for url in discovered_links if is_priority_url(url)]
    priority_links.sort(key=score_url_relevance, reverse=True)

    other_links = [url for url in discovered_links if not is_priority_url(url)]

    candidate_urls = priority_links + other_links
    logger.info(f"[INFO] Relevant pages discovered: {len(priority_links)} (Total internal: {len(discovered_links)})")

    # Step 3: Crawl high-priority internal pages up to max_pages
    pages_to_crawl = max_pages - 1  # Homepage is already 1

    for url in candidate_urls:
        if len(crawled_pages) >= max_pages:
            break

        canonical_check = url.rstrip("/")
        if canonical_check in visited_urls or f"{canonical_check}/" in visited_urls:
            continue

        visited_urls.add(canonical_check)
        visited_urls.add(f"{canonical_check}/")

        page_html, page_status, page_err = fetch_page_content(browser, url)

        if page_err or not page_html:
            logger.warning(f"[WARNING] Could not fetch {url}: {page_err or 'Empty response'}")
            continue

        page_data = extract_page_data(page_html, url, clean_domain)
        if page_data["cleaned_text"]:
            crawled_pages.append(page_data)
            all_emails.update(page_data["emails"])
            all_linkedin_urls.update(page_data["linkedin_urls"])

    logger.info(f"[INFO] Content cleaned across {len(crawled_pages)} pages")
    logger.info(f"[INFO] Emails discovered: {len(all_emails)}")
    logger.info(f"[INFO] LinkedIn URLs discovered: {len(all_linkedin_urls)}")

    combined_text = "\n\n".join(
        f"--- Page: {p['url']} ---\n{p['cleaned_text']}" for p in crawled_pages
    )

    return {
        "domain": clean_domain,
        "pages": crawled_pages,
        "combined_cleaned_text": combined_text,
        "discovered_emails": sorted(all_emails),
        "discovered_linkedin_urls": sorted(all_linkedin_urls),
        "error": None,
    }
