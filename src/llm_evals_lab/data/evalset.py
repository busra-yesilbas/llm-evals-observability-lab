"""
Eval set builder for the LLM Evals Lab.

Loads and saves evaluation examples. Also provides factory methods
to generate a synthetic evaluation set from the corpus.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from llm_evals_lab.schemas import Difficulty, EvalExample, QuestionCategory

logger = logging.getLogger(__name__)


class EvalSetLoader:
    """Loads and saves evaluation sets from the eval/ directory."""

    def __init__(self, eval_dir: Path) -> None:
        self.eval_dir = eval_dir

    def load(self, filename: str = "eval_set.json") -> list[EvalExample]:
        """Load the evaluation set from disk."""
        path = self.eval_dir / filename
        if not path.exists():
            logger.warning("Eval set not found: %s", path)
            return []
        with path.open("r", encoding="utf-8") as fh:
            records = json.load(fh)
        examples = [EvalExample(**r) for r in records]
        logger.info("Loaded %d eval examples from %s", len(examples), path)
        return examples

    def save(
        self, examples: list[EvalExample], filename: str = "eval_set.json"
    ) -> Path:
        """Persist eval examples to disk."""
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        path = self.eval_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                [e.model_dump() for e in examples],
                fh,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        logger.info("Saved %d eval examples to %s", len(examples), path)
        return path


def build_synthetic_eval_set() -> list[EvalExample]:
    """
    Build a synthetic evaluation set for the NovaSaaS support knowledge base.

    Covers six question categories across three difficulty levels.
    Designed to test common RAG failure modes:
    - Weak retrieval (vague/multi-hop questions)
    - Missing citations
    - Hallucination risk (questions with tempting wrong answers)
    - Correct abstention (unanswerable questions)
    """
    examples: list[EvalExample] = [
        # ── Factual / Direct lookup ───────────────────────────────────────────
        EvalExample(
            example_id="eval_001",
            question="What is the standard refund window for NovaSaaS subscriptions?",
            reference_answer="NovaSaaS offers a 30-day money-back guarantee for all subscription plans. Refund requests must be submitted through the billing portal within 30 days of the original charge.",
            expected_doc_ids=["doc_billing_001"],
            expected_key_points=["30-day", "money-back guarantee", "billing portal"],
            difficulty=Difficulty.EASY,
            category=QuestionCategory.FACTUAL,
        ),
        EvalExample(
            example_id="eval_002",
            question="How many users can be added under the Starter plan?",
            reference_answer="The Starter plan supports up to 5 user seats. Additional seats can be purchased at $15 per seat per month. Organizations exceeding 5 seats are automatically prompted to upgrade to the Growth plan.",
            expected_doc_ids=["doc_pricing_001"],
            expected_key_points=["5 user seats", "$15 per seat", "Growth plan"],
            difficulty=Difficulty.EASY,
            category=QuestionCategory.FACTUAL,
        ),
        EvalExample(
            example_id="eval_003",
            question="What SSO providers does NovaSaaS support?",
            reference_answer="NovaSaaS supports SAML 2.0-based SSO with Okta, Azure Active Directory, Google Workspace, and OneLogin. SCIM provisioning is available on the Business and Enterprise plans.",
            expected_doc_ids=["doc_security_001"],
            expected_key_points=["SAML 2.0", "Okta", "Azure AD", "Google Workspace", "SCIM"],
            difficulty=Difficulty.EASY,
            category=QuestionCategory.FACTUAL,
        ),
        EvalExample(
            example_id="eval_004",
            question="What is the maximum file size for attachments in NovaSaaS?",
            reference_answer="NovaSaaS supports file attachments up to 50 MB per file. The total storage quota depends on the subscription plan: 10 GB for Starter, 100 GB for Growth, and unlimited for Enterprise.",
            expected_doc_ids=["doc_storage_001"],
            expected_key_points=["50 MB", "10 GB", "100 GB", "unlimited"],
            difficulty=Difficulty.EASY,
            category=QuestionCategory.FACTUAL,
        ),
        EvalExample(
            example_id="eval_005",
            question="What is the uptime SLA for the Enterprise plan?",
            reference_answer="The Enterprise plan includes a 99.9% uptime SLA with dedicated support. Customers on the Enterprise plan also receive a 24/7 dedicated support channel and a guaranteed response time of 1 hour for critical issues.",
            expected_doc_ids=["doc_sla_001"],
            expected_key_points=["99.9%", "dedicated support", "24/7", "1 hour"],
            difficulty=Difficulty.EASY,
            category=QuestionCategory.FACTUAL,
        ),
        # ── Policy interpretation ─────────────────────────────────────────────
        EvalExample(
            example_id="eval_006",
            question="Can a customer downgrade from Business to Starter mid-cycle?",
            reference_answer="Yes, plan downgrades are permitted but take effect at the next billing cycle. Customers will retain access to Business-tier features until the current cycle ends. Prorated refunds are not issued for downgrades.",
            expected_doc_ids=["doc_billing_001", "doc_pricing_001"],
            expected_key_points=["next billing cycle", "no prorated refund", "retain access"],
            difficulty=Difficulty.MEDIUM,
            category=QuestionCategory.POLICY,
        ),
        EvalExample(
            example_id="eval_007",
            question="What happens to user data after an account is deleted?",
            reference_answer="Upon account deletion, NovaSaaS retains user data for 30 days in a soft-deleted state to allow for recovery. After 30 days, all data is permanently purged from production systems. Compliance archives are retained for 7 years per regulatory requirements.",
            expected_doc_ids=["doc_data_policy_001"],
            expected_key_points=["30 days", "soft-deleted", "permanently purged", "7 years compliance"],
            difficulty=Difficulty.MEDIUM,
            category=QuestionCategory.POLICY,
        ),
        EvalExample(
            example_id="eval_008",
            question="Is NovaSaaS GDPR compliant? What mechanisms are provided for data subject requests?",
            reference_answer="NovaSaaS is GDPR compliant. It provides a Data Subject Request portal for access, correction, and erasure requests. Requests are processed within 30 days. A Data Processing Agreement (DPA) is available for Enterprise customers.",
            expected_doc_ids=["doc_compliance_001"],
            expected_key_points=["GDPR", "Data Subject Request", "30 days", "DPA", "erasure"],
            difficulty=Difficulty.MEDIUM,
            category=QuestionCategory.POLICY,
        ),
        # ── Multi-hop / synthesis ─────────────────────────────────────────────
        EvalExample(
            example_id="eval_009",
            question="A company with 12 employees needs SSO and 99.9% uptime. Which plan should they choose and what will it cost approximately?",
            reference_answer="A company with 12 employees needing SSO and 99.9% uptime should choose the Business plan, which includes SSO and supports up to 50 seats. The Business plan is priced at $49 per seat per month, totaling approximately $588/month for 12 seats. The Enterprise plan also meets these requirements with additional SLA guarantees.",
            expected_doc_ids=["doc_pricing_001", "doc_security_001", "doc_sla_001"],
            expected_key_points=["Business plan", "SSO", "99.9%", "$49 per seat", "50 seats"],
            difficulty=Difficulty.HARD,
            category=QuestionCategory.MULTI_HOP,
        ),
        EvalExample(
            example_id="eval_010",
            question="What are the steps to migrate data from a legacy CRM into NovaSaaS, and what data formats are supported?",
            reference_answer="NovaSaaS supports data migration from CSV, JSON, and Excel (XLSX) formats. The migration wizard in Settings > Data Import guides users through field mapping, validation, and import. For large datasets (>100k records), the Support team provides a managed migration service at no additional cost on Business and Enterprise plans.",
            expected_doc_ids=["doc_onboarding_001", "doc_integrations_001"],
            expected_key_points=["CSV", "JSON", "XLSX", "migration wizard", "field mapping", "100k records"],
            difficulty=Difficulty.HARD,
            category=QuestionCategory.MULTI_HOP,
        ),
        EvalExample(
            example_id="eval_011",
            question="How does NovaSaaS handle API rate limiting, and what should a developer do if they hit the limit?",
            reference_answer="NovaSaaS enforces API rate limits of 1,000 requests per minute for the Starter plan and 10,000 requests per minute for Growth and above. When the limit is exceeded, the API returns HTTP 429 Too Many Requests with a Retry-After header. Developers should implement exponential backoff. Permanent limit increases can be requested through the developer portal.",
            expected_doc_ids=["doc_api_001"],
            expected_key_points=["1,000 per minute", "10,000 per minute", "HTTP 429", "Retry-After", "exponential backoff"],
            difficulty=Difficulty.HARD,
            category=QuestionCategory.MULTI_HOP,
        ),
        # ── Comparison ────────────────────────────────────────────────────────
        EvalExample(
            example_id="eval_012",
            question="What are the main differences between the Growth and Business plans?",
            reference_answer="The Growth plan supports up to 25 seats at $29/seat/month and includes basic analytics, 100 GB storage, and email support. The Business plan supports up to 50 seats at $49/seat/month and adds SSO, advanced analytics, audit logs, custom roles, and priority support. Both plans include the core platform features.",
            expected_doc_ids=["doc_pricing_001"],
            expected_key_points=["25 seats", "50 seats", "$29", "$49", "SSO", "audit logs", "custom roles"],
            difficulty=Difficulty.MEDIUM,
            category=QuestionCategory.COMPARISON,
        ),
        EvalExample(
            example_id="eval_013",
            question="How does NovaSaaS's webhook system compare to its native integrations in terms of flexibility and setup complexity?",
            reference_answer="Native integrations offer one-click setup and automatic field mapping but are limited to the 50+ supported applications. Webhooks provide custom event-driven integration with any HTTP endpoint, offering greater flexibility. However, webhooks require developer configuration including payload parsing, signature verification, and error handling. For non-technical teams, native integrations are recommended.",
            expected_doc_ids=["doc_integrations_001", "doc_api_001"],
            expected_key_points=["one-click", "50+ apps", "HTTP endpoint", "signature verification", "developer configuration"],
            difficulty=Difficulty.HARD,
            category=QuestionCategory.COMPARISON,
        ),
        # ── Ambiguous questions ───────────────────────────────────────────────
        EvalExample(
            example_id="eval_014",
            question="How do I cancel?",
            reference_answer="To cancel your NovaSaaS subscription, navigate to Settings > Billing > Subscription and click 'Cancel Subscription'. Cancellations take effect at the end of the current billing period. If you are within the 30-day refund window, you may request a full refund through the billing portal.",
            expected_doc_ids=["doc_billing_001"],
            expected_key_points=["Settings > Billing", "end of billing period", "30-day refund"],
            difficulty=Difficulty.EASY,
            category=QuestionCategory.AMBIGUOUS,
            notes="Intentionally vague — tests whether the RAG system resolves to 'cancel subscription'",
        ),
        EvalExample(
            example_id="eval_015",
            question="Is my data safe?",
            reference_answer="NovaSaaS employs multiple layers of security: AES-256 encryption at rest, TLS 1.2+ in transit, SOC 2 Type II certification, GDPR compliance, regular third-party penetration testing, and role-based access control. Enterprise customers also receive a dedicated security review and BAA for HIPAA use cases.",
            expected_doc_ids=["doc_security_001", "doc_compliance_001"],
            expected_key_points=["AES-256", "TLS", "SOC 2", "GDPR", "penetration testing"],
            difficulty=Difficulty.MEDIUM,
            category=QuestionCategory.AMBIGUOUS,
            notes="Vague question — tests breadth of retrieval and summarization ability",
        ),
        # ── Unanswerable / insufficient context ──────────────────────────────
        EvalExample(
            example_id="eval_016",
            question="What is the exact datacenter IP address range used by NovaSaaS?",
            reference_answer="This information is not available in the public documentation. IP allowlisting ranges are provided to Enterprise customers through a private security briefing. Contact your account manager.",
            expected_doc_ids=[],
            expected_key_points=["not in documentation", "Enterprise", "account manager"],
            difficulty=Difficulty.HARD,
            category=QuestionCategory.UNANSWERABLE,
            is_answerable=False,
            notes="Should trigger graceful abstention",
        ),
        EvalExample(
            example_id="eval_017",
            question="What is the personal phone number for NovaSaaS's CEO?",
            reference_answer="This information is not available in NovaSaaS's documentation. Contact information for the executive team is not publicly disclosed.",
            expected_doc_ids=[],
            expected_key_points=["not available", "not disclosed"],
            difficulty=Difficulty.EASY,
            category=QuestionCategory.UNANSWERABLE,
            is_answerable=False,
            notes="Should trigger abstention — completely out-of-scope",
        ),
        EvalExample(
            example_id="eval_018",
            question="Can I use NovaSaaS to store classified government data under ITAR regulations?",
            reference_answer="The provided documentation does not address ITAR compliance. NovaSaaS is SOC 2 and GDPR compliant, but specific regulatory frameworks such as ITAR, FedRAMP, or FISMA are not mentioned. You should contact the NovaSaaS Enterprise sales team for a compliance assessment.",
            expected_doc_ids=["doc_compliance_001"],
            expected_key_points=["not addressed", "contact sales", "SOC 2", "ITAR"],
            difficulty=Difficulty.HARD,
            category=QuestionCategory.UNANSWERABLE,
            is_answerable=False,
            notes="Partially answerable — system can state what IS in docs but should acknowledge gaps",
        ),
        # ── Additional factual ────────────────────────────────────────────────
        EvalExample(
            example_id="eval_019",
            question="Does NovaSaaS offer a free trial?",
            reference_answer="Yes, NovaSaaS offers a 14-day free trial of the Growth plan with no credit card required. All features of the Growth plan are available during the trial period. After 14 days, the account is automatically downgraded to the Starter free tier unless a paid plan is selected.",
            expected_doc_ids=["doc_pricing_001"],
            expected_key_points=["14-day", "no credit card", "Growth plan features", "Starter free tier"],
            difficulty=Difficulty.EASY,
            category=QuestionCategory.FACTUAL,
        ),
        EvalExample(
            example_id="eval_020",
            question="What notification channels does NovaSaaS support for alerts and workflows?",
            reference_answer="NovaSaaS supports email, Slack, Microsoft Teams, and SMS (Enterprise only) as notification channels for alerts and workflow automations. Webhooks can also be configured as custom notification endpoints. Notification preferences are managed per-user in Profile > Notifications.",
            expected_doc_ids=["doc_integrations_001", "doc_onboarding_001"],
            expected_key_points=["email", "Slack", "Teams", "SMS", "webhooks"],
            difficulty=Difficulty.MEDIUM,
            category=QuestionCategory.FACTUAL,
        ),
    ]
    return examples
