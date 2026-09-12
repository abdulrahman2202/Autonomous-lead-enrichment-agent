"""Pydantic schemas for the Autonomous Lead Enrichment Agent.

Defines the structured output models for company intelligence and leadership
members with strict type validation.
"""

from pydantic import BaseModel, Field


class LeadershipMember(BaseModel):
    """Represents a key executive or leadership team member of a company."""

    name: str = Field(description="Full name of the executive or leader")
    role: str = Field(description="Role, title, or position within the company")
    linkedin_url: str | None = Field(
        default=None,
        description="Legitimate LinkedIn profile URL if identified in evidence",
    )


class CompanyData(BaseModel):
    """Structured company intelligence model extracted from public web sources."""

    domain: str = Field(description="Normalized company domain")
    company_overview: str = Field(
        description="Concise two-sentence summary of what the company does"
    )
    target_audience: str = Field(
        description="Ideal Customer Profile (ICP) and primary target audience"
    )
    contact_points: list[str] = Field(
        default_factory=list,
        description="Public/generic contact emails found in website evidence",
    )
    leadership: list[LeadershipMember] = Field(
        default_factory=list,
        description="Verified leadership team members discovered on the website",
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Extraction confidence score strictly bounded between 0.0 and 1.0",
    )


def create_fallback_company_data(
    domain: str,
    error_message: str = "",
    contact_points: list[str] | None = None,
) -> CompanyData:
    """Creates a safe fallback CompanyData record when extraction or validation fails.

    Preserves verified contact points discovered by the crawler while assigning zero
    confidence to unverified fields.

    Args:
        domain: The domain that failed processing.
        error_message: Optional reason for the failure.
        contact_points: Optional verified contact emails extracted from the website.

    Returns:
        A valid CompanyData instance with zero confidence score.
    """
    overview_text = (
        f"Extraction could not be completed for {domain}. {error_message}".strip()
    )
    return CompanyData(
        domain=domain,
        company_overview=overview_text,
        target_audience="Unavailable due to crawl or extraction limitation",
        contact_points=contact_points or [],
        leadership=[],
        confidence_score=0.0,
    )

