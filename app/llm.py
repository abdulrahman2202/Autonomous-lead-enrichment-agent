"""Ollama Gemma 3 extraction and structured Pydantic validation module.

Interacts with a local Ollama instance running gemma3:latest to perform
anti-hallucinatory company intelligence extraction with strict schema validation.
"""

import json
import logging
from typing import Any
import ollama
from pydantic import ValidationError

from app.schemas import CompanyData, create_fallback_company_data
from app.utils import clean_and_parse_json

logger = logging.getLogger("lead_enrichment")

DEFAULT_MODEL = "gemma3:latest"

EXTRACTION_SYSTEM_PROMPT = """You are a company intelligence extraction system.
Your task is to analyze extracted public website evidence and produce structured, factual company intelligence.

STRICT ANTI-HALLUCINATION RULES:
1. Rely EXCLUSIVELY on the provided website evidence and verified contact evidence.
2. Absolutely DO NOT use external knowledge, training memory, or assumptions not grounded in the text.
3. Absolutely DO NOT invent people, executives, emails, or URLs.
4. Only include leadership members if their name and role are explicitly mentioned in the provided text.
5. Only include emails that are present in the provided contact evidence or text.
6. Only include LinkedIn URLs that are present in the provided evidence.
7. If any information (e.g. leadership, contact emails) is absent or incomplete in the evidence, return an empty list [] or null.
8. The confidence_score must be a float strictly between 0.0 and 1.0 reflecting the completeness and factual grounding of the evidence.
9. Output ONLY a valid JSON object matching the required schema. No introductory text, no conversational commentary, and no markdown outside the JSON."""


def build_extraction_prompt(
    domain: str,
    optimized_context: str,
    discovered_emails: list[str],
    discovered_linkedin_urls: list[str],
) -> str:
    """Constructs the user prompt containing extracted evidence and schema expectations.

    Args:
        domain: Company domain.
        optimized_context: Cleaned, truncated website text.
        discovered_emails: Verified public email addresses found on the site.
        discovered_linkedin_urls: Legitimate LinkedIn URLs found on the site.

    Returns:
        Formatted prompt string.
    """
    emails_str = json.dumps(discovered_emails) if discovered_emails else "[] (None found on site)"
    linkedin_str = (
        json.dumps(discovered_linkedin_urls)
        if discovered_linkedin_urls
        else "[] (None found on site)"
    )

    return f"""Target Company Domain: {domain}

VERIFIED CONTACT EVIDENCE:
- Discovered Public Emails: {emails_str}
- Discovered LinkedIn URLs: {linkedin_str}

WEBSITE TEXT EVIDENCE:
{optimized_context}

EXPECTED OUTPUT JSON SCHEMA:
{{
  "domain": "{domain}",
  "company_overview": "<Approx two concise sentences summarizing what the company does and its core offering>",
  "target_audience": "<Primary Ideal Customer Profile (ICP) and who the product is built for>",
  "contact_points": [
    "<generic or public emails strictly from the verified contact evidence above, or empty list if none>"
  ],
  "leadership": [
    {{
      "name": "<Full name of executive/leader explicitly identified in the text>",
      "role": "<Title/position>",
      "linkedin_url": "<Matching LinkedIn URL from verified evidence, or null>"
    }}
  ],
  "confidence_score": <float between 0.0 and 1.0>
}}

Extract the company intelligence now and output strictly the JSON object:"""


def extract_company_intelligence(
    domain: str,
    optimized_context: str,
    discovered_emails: list[str],
    discovered_linkedin_urls: list[str],
    model: str = DEFAULT_MODEL,
) -> CompanyData:
    """Invokes Ollama Gemma 3 to extract structured company intelligence and validates output.

    Args:
        domain: Company domain.
        optimized_context: Cleaned website text context.
        discovered_emails: Discovered email evidence.
        discovered_linkedin_urls: Discovered LinkedIn evidence.
        model: Ollama model tag. Defaults to 'gemma3:latest'.

    Returns:
        Validated CompanyData instance or safe fallback record on failure.
    """
    logger.info(f"[INFO] Sending context to Gemma 3 ({model})")

    user_prompt = build_extraction_prompt(
        domain=domain,
        optimized_context=optimized_context,
        discovered_emails=discovered_emails,
        discovered_linkedin_urls=discovered_linkedin_urls,
    )

    try:
        response: Any = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
            options={
                "temperature": 0.1,
                "top_p": 0.9,
            },
        )
    except Exception as e:
        err_msg = f"Ollama extraction failed for {domain}: {str(e)}"
        logger.error(f"[ERROR] {err_msg}")
        return create_fallback_company_data(domain, error_message=err_msg)

    # Safely retrieve content across different ollama python client versions
    raw_content = ""
    if hasattr(response, "message") and hasattr(response.message, "content"):
        raw_content = response.message.content or ""
    elif isinstance(response, dict) and "message" in response:
        raw_content = response["message"].get("content", "")
    else:
        raw_content = str(response)

    logger.info("[INFO] LLM response received")

    # Clean and parse JSON
    parsed_json = clean_and_parse_json(raw_content)

    if not parsed_json:
        err_msg = f"Could not parse valid JSON from LLM response for {domain}"
        logger.warning(f"[WARNING] {err_msg}. Raw response preview: {raw_content[:200]}")
        return create_fallback_company_data(domain, error_message=err_msg)

    # Strict Pydantic Schema Validation (no silent clamping)
    try:
        # Enforce target domain in the parsed dictionary
        parsed_json["domain"] = domain
        company_data = CompanyData.model_validate(parsed_json)
        logger.info("[INFO] Pydantic validation successful")
        return company_data
    except ValidationError as ve:
        err_msg = f"Pydantic validation error for {domain}: {ve.errors()}"
        logger.warning(f"[WARNING] {err_msg}")
        return create_fallback_company_data(
            domain, error_message=f"Schema validation failed: {str(ve)}"
        )
    except Exception as ex:
        err_msg = f"Unexpected error during model validation for {domain}: {str(ex)}"
        logger.warning(f"[WARNING] {err_msg}")
        return create_fallback_company_data(domain, error_message=err_msg)
