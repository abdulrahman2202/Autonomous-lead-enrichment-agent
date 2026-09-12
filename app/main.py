"""Main orchestration pipeline for the Autonomous Lead Enrichment Agent.

Iterates over target company domains, executes Playwright browser crawling,
cleans webpage content, extracts verified contact points, sends optimized context
to Ollama Gemma 3, validates structured results using Pydantic, and writes output
to output/output.json.
"""

import json
import logging
import sys
from pathlib import Path

from app.browser import get_browser_session
from app.crawler import crawl_domain
from app.llm import extract_company_intelligence
from app.schemas import CompanyData, create_fallback_company_data
from app.utils import optimize_context

# Target domains specified for lead enrichment
COMPANIES = [
    "postman.com",
    "supabase.com",
    "vapi.ai",
]


def setup_logger() -> logging.Logger:
    """Configures clean console logging matching project demonstration guidelines."""
    logger = logging.getLogger("lead_enrichment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def run_pipeline(
    companies: list[str] = COMPANIES,
    output_path: Path | None = None,
    max_pages_per_domain: int = 6,
) -> list[CompanyData]:
    """Executes the lead enrichment pipeline across target company domains.

    Args:
        companies: List of domain strings to enrich.
        output_path: Path to the destination JSON file. Defaults to output/output.json.
        max_pages_per_domain: Maximum pages crawled per company. Defaults to 6.

    Returns:
        List of validated CompanyData objects.
    """
    logger = setup_logger()
    logger.info("==================================================")
    logger.info("Starting Autonomous Lead Enrichment Agent Pipeline")
    logger.info("Target Companies: " + ", ".join(companies))
    logger.info("==================================================")

    if output_path is None:
        project_root = Path(__file__).resolve().parent.parent
        output_path = project_root / "output" / "output.json"

    results: list[CompanyData] = []

    # Single browser session reused across company crawls for efficiency and safety
    with get_browser_session(headless=True) as browser:
        for domain in companies:
            logger.info("--------------------------------------------------")
            try:
                # 1. Crawl domain and discover internal pages
                crawl_result = crawl_domain(
                    domain=domain,
                    browser=browser,
                    max_pages=max_pages_per_domain,
                )

                if crawl_result.get("error") and not crawl_result.get("pages"):
                    logger.warning(
                        f"[WARNING] Crawling yielded no accessible pages for {domain}. Generating fallback."
                    )
                    fallback = create_fallback_company_data(
                        domain=domain,
                        error_message=crawl_result.get("error", "No pages crawled"),
                    )
                    results.append(fallback)
                    continue

                # 2. Optimize context for Gemma 3
                optimized_context = optimize_context(crawl_result.get("pages", []))

                # 3. Extract company intelligence with Ollama Gemma 3 and validate with Pydantic
                company_data = extract_company_intelligence(
                    domain=domain,
                    optimized_context=optimized_context,
                    discovered_emails=crawl_result.get("discovered_emails", []),
                    discovered_linkedin_urls=crawl_result.get("discovered_linkedin_urls", []),
                )

                results.append(company_data)

            except Exception as e:
                logger.error(f"[ERROR] Unexpected pipeline error while processing {domain}: {str(e)}")
                fallback = create_fallback_company_data(
                    domain=domain,
                    error_message=f"Pipeline exception: {str(e)}",
                )
                results.append(fallback)

    # 4. Save results to output/output.json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_results = [result.model_dump() for result in results]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialized_results, f, indent=2, ensure_ascii=False)

    logger.info("==================================================")
    logger.info(f"[INFO] Pipeline complete! Output successfully written to: {output_path}")
    logger.info(f"[INFO] Processed {len(results)} companies.")
    logger.info("==================================================")

    return results


def main() -> None:
    """Entry point for command-line execution."""
    run_pipeline()


if __name__ == "__main__":
    main()
