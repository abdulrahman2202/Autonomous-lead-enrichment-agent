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
from app.utils import extract_and_parse_json

logger = logging.getLogger("lead_enrichment")

DEFAULT_MODEL = "gemma3:latest"

EXTRACTION_SYSTEM_PROMPT = """You are a company intelligence extraction system.
Analyze the supplied website evidence and extract structured company data.

RULES:
1. Return EXACTLY ONE JSON object.
2. Return JSON ONLY.
3. No Markdown, no code fences (do not use ```json), and no explanations.
4. NEVER return {}.
5. Always return all six required top-level fields:
   - "domain": string
   - "company_overview": string
   - "target_audience": string
   - "contact_points": list of strings
   - "leadership": list of objects
   - "confidence_score": float (between 0.0 and 1.0)
6. If information is unavailable from the supplied evidence:
   - company_overview -> ""
   - target_audience -> ""
   - contact_points -> []
   - leadership -> []
   - linkedin_url -> null
   - confidence_score -> lower value (0.0 to 0.5) reflecting missing evidence
7. Do not omit any fields.
8. Use ONLY supplied website evidence. Never invent people, roles, emails, or LinkedIn URLs.
9. Never assign a company LinkedIn URL (/company/...) to an individual leader. Only use individual /in/ URLs from verified evidence, or null."""


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
    emails_json = json.dumps(discovered_emails) if discovered_emails else "[]"
    linkedin_json = (
        json.dumps(discovered_linkedin_urls) if discovered_linkedin_urls else "[]"
    )

    return f"""Target Company Domain: {domain}

VERIFIED CONTACT EVIDENCE (DISCOVERED ON WEBSITE):
- Discovered Emails: {emails_json}
- Discovered LinkedIn URLs: {linkedin_json}

WEBSITE CONTENT:
{optimized_context}

Return a single JSON object with this exact structure:
{{
  "domain": "{domain}",
  "company_overview": "<two concise sentences about what the company does and its core offering>",
  "target_audience": "<primary Ideal Customer Profile and target audience>",
  "contact_points": {emails_json},
  "leadership": [
    {{
      "name": "<name of executive/leader explicitly identified in text>",
      "role": "<role/title explicitly identified in text>",
      "linkedin_url": null
    }}
  ],
  "confidence_score": 0.9
}}

CRITICAL INSTRUCTIONS:
- Do NOT return {{}}.
- Do NOT omit any fields.
- If no leadership members are explicitly named in the website content, set "leadership": [].
- Never assign a company LinkedIn URL (/company/...) to an individual person; use null instead.
- Output raw JSON only."""


def normalize_extracted_data(
    parsed_json: dict,
    domain: str,
    discovered_emails: list[str],
    discovered_linkedin_urls: list[str],
) -> dict:
    """Normalizes raw LLM output, enforcing deterministic evidence and schema integrity.

    Args:
        parsed_json: Raw dictionary parsed from the LLM output.
        domain: Target company domain.
        discovered_emails: Authoritative emails extracted by Python crawler.
        discovered_linkedin_urls: Authoritative LinkedIn URLs extracted by Python crawler.

    Returns:
        Normalized dictionary ready for strict Pydantic validation.
    """
    normalized = dict(parsed_json)

    # 1. Enforce domain
    normalized["domain"] = domain

    # 2. Text fields: ensure string type and clean whitespace
    raw_overview = normalized.get("company_overview")
    normalized["company_overview"] = str(raw_overview).strip() if raw_overview is not None else ""

    raw_audience = normalized.get("target_audience")
    normalized["target_audience"] = str(raw_audience).strip() if raw_audience is not None else ""

    # 3. Deterministic Contact Points: Python-discovered emails are strictly authoritative
    # Discard any email invented by the LLM that wasn't actually discovered on the website
    valid_crawler_emails = set(discovered_emails)
    normalized["contact_points"] = sorted(valid_crawler_emails)

    # 4. Leadership & LinkedIn URL validation
    # Differentiate individual profiles (/in/) from company pages (/company/)
    valid_individual_profiles = {
        u.rstrip("/").lower(): u.rstrip("/")
        for u in discovered_linkedin_urls
        if "/in/" in u.lower()
    }

    normalized_leadership = []
    raw_leadership = normalized.get("leadership")
    if isinstance(raw_leadership, list):
        for member in raw_leadership:
            if not isinstance(member, dict):
                continue
            name = str(member.get("name") or "").strip()
            role = str(member.get("role") or "").strip()
            if not name or not role:
                continue

            raw_li = member.get("linkedin_url")
            clean_li = None
            if isinstance(raw_li, str) and raw_li.strip():
                cand_li = raw_li.strip().rstrip("/")
                # Strictly reject company URLs for individuals
                if "/company/" in cand_li.lower():
                    clean_li = None
                elif "/in/" in cand_li.lower():
                    # Must match an actual individual profile discovered on the site
                    matched = valid_individual_profiles.get(cand_li.lower())
                    clean_li = matched
                else:
                    clean_li = None

            normalized_leadership.append(
                {
                    "name": name,
                    "role": role,
                    "linkedin_url": clean_li,
                }
            )
    normalized["leadership"] = normalized_leadership

    # 5. Confidence score normalization if omitted
    if "confidence_score" not in normalized or normalized["confidence_score"] is None:
        if normalized["company_overview"] and normalized["target_audience"]:
            normalized["confidence_score"] = 0.8
        else:
            normalized["confidence_score"] = 0.0

    return normalized


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
                "num_ctx": 8192,
            },
        )
    except Exception as e:
        err_msg = f"Ollama extraction failed for {domain}: {str(e)}"
        logger.error(f"[ERROR] {err_msg}")
        return create_fallback_company_data(
            domain, error_message=err_msg, contact_points=discovered_emails
        )

    # Safely retrieve content across different ollama python client versions
    raw_content = ""
    if hasattr(response, "message") and hasattr(response.message, "content"):
        raw_content = response.message.content or ""
    elif isinstance(response, dict) and "message" in response:
        raw_content = response["message"].get("content", "")
    else:
        raw_content = str(response)

    logger.info("[INFO] LLM response received")

    # Step 1: Clean and parse JSON using robust extraction pipeline
    parsed_json = extract_and_parse_json(raw_content)

    if parsed_json is None:
        err_msg = f"Could not parse valid JSON from LLM response for {domain}"
        logger.warning(
            f"[WARNING] {err_msg}. Raw response preview: {raw_content[:200]}"
        )
        return create_fallback_company_data(
            domain, error_message=err_msg, contact_points=discovered_emails
        )

    # Step 2: Handle empty dict {} or completely unpopulated response
    if not parsed_json or (
        not parsed_json.get("company_overview")
        and not parsed_json.get("target_audience")
    ):
        err_msg = f"LLM returned an empty or unpopulated object for {domain}"
        logger.warning(f"[WARNING] {err_msg}. Applying safe fallback.")
        return create_fallback_company_data(
            domain,
            error_message="LLM returned an empty object with no company intelligence.",
            contact_points=discovered_emails,
        )

    # Step 3: Normalize fields and enforce deterministic evidence
    normalized_data = normalize_extracted_data(
        parsed_json=parsed_json,
        domain=domain,
        discovered_emails=discovered_emails,
        discovered_linkedin_urls=discovered_linkedin_urls,
    )

    # Step 4: Strict Pydantic Schema Validation (no silent clamping of confidence score)
    try:
        company_data = CompanyData.model_validate(normalized_data)
        logger.info("[INFO] Pydantic validation successful")
        return company_data
    except ValidationError as ve:
        err_msg = f"Pydantic validation error for {domain}: {ve.errors()}"
        logger.warning(f"[WARNING] {err_msg}")
        return create_fallback_company_data(
            domain,
            error_message=f"Schema validation failed: {str(ve)}",
            contact_points=discovered_emails,
        )
    except Exception as ex:
        err_msg = f"Unexpected error during model validation for {domain}: {str(ex)}"
        logger.warning(f"[WARNING] {err_msg}")
        return create_fallback_company_data(
            domain,
            error_message=err_msg,
            contact_points=discovered_emails,
        )
