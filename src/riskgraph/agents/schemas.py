"""Agent output schemas. IncidentReport is SPEC §10.3; the rest are internal node outputs."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

RootCause = Literal[
    "position_change", "market_move", "bad_market_data", "no_true_breach", "unknown"
]
Action = Literal["escalate_to_risk_manager", "route_to_data_ops", "no_action"]
Specialist = Literal["attribution", "data_quality", "policy"]


class Breach(BaseModel):
    scope: str
    metric: str
    value: float
    limit: float
    utilization: float


class Evidence(BaseModel):
    claim: str = Field(description="One factual statement the value supports.")
    value: float = Field(description="Number copied exactly from the tool result.")
    unit: str = Field(description="Unit as in the tool result, e.g. USD, fraction, bp.")
    result_id: str = Field(description="result_id of the tool result the value comes from.")


class Citation(BaseModel):
    doc_id: str = Field(description="Document ID, e.g. MRLP, MDCP, MAR, CRE.")
    section_id: str = Field(description="Section ID exactly as retrieved, e.g. MRLP-6.1.")


class IncidentReport(BaseModel):
    """SPEC §10.3."""

    incident_id: str
    as_of_date: date
    breach: Breach
    root_cause: RootCause
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence]
    recommended_action: Action
    policy_citations: list[Citation]
    draft_note: str


class ReportDraft(BaseModel):
    """What the writer (or the baseline agent) generates; intake supplies the rest."""

    root_cause: RootCause
    confidence: float = Field(ge=0, le=1, description="Probability the root cause is right.")
    evidence: list[Evidence]
    recommended_action: Action
    policy_citations: list[Citation]
    draft_note: str = Field(description="Escalation note for a risk manager, in Markdown.")


class Task(BaseModel):
    agent: Specialist
    question: str


class Plan(BaseModel):
    """Supervisor output: which specialists to call and what to ask each."""

    tasks: list[Task]


class Findings(BaseModel):
    """Specialist output."""

    summary: str = Field(description="What the tool results show, in 2-5 sentences.")
    evidence: list[Evidence]
    citations: list[Citation] = Field(
        default_factory=list, description="Policy agent only: sections that apply."
    )


class NoteReview(BaseModel):
    """Critic check 4: is the note clear, correctly prioritized, and actionable?"""

    ok: bool
    feedback: str = Field(description="Specific fixes if not ok, else empty.")
