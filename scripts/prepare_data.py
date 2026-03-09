"""
prepare_data.py — Corpus preparation pipeline.

Creates the synthetic NovaSaaS knowledge base, chunks all documents,
and saves processed artifacts to data/raw/ and data/processed/.

Run from project root:
    python scripts/prepare_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_evals_lab.config import load_config
from llm_evals_lab.data.chunking import Chunker
from llm_evals_lab.data.loader import CorpusLoader
from llm_evals_lab.schemas import SourceDocument
from llm_evals_lab.utils import setup_logging


def build_corpus() -> list[SourceDocument]:
    """Return the synthetic NovaSaaS support knowledge base corpus."""
    return [
        SourceDocument(
            doc_id="doc_billing_001",
            title="Billing & Subscription Management",
            source_type="policy",
            category="billing",
            full_text="""
NovaSaaS Billing & Subscription Management

Overview
NovaSaaS uses a seat-based monthly or annual billing model. All plans are billed in USD. Annual subscriptions receive a 20% discount compared to monthly rates.

Refund Policy
NovaSaaS offers a 30-day money-back guarantee for all subscription plans. Refund requests must be submitted through the billing portal within 30 days of the original charge. After 30 days, refunds are not available except at the discretion of the NovaSaaS billing team for exceptional circumstances. Annual subscriptions that are cancelled after 30 days receive prorated credit for unused months, applied to future invoices.

Plan Changes and Upgrades
Customers may upgrade their plan at any time. Upgrades take effect immediately and are billed prorated for the remainder of the current billing cycle. Plan downgrades are permitted but take effect at the next billing cycle. Customers will retain access to higher-tier features until the current cycle ends. Prorated refunds are not issued for downgrades.

Cancellation
To cancel your NovaSaaS subscription, navigate to Settings > Billing > Subscription and click 'Cancel Subscription'. Cancellations take effect at the end of the current billing period. Your account data is retained for 30 days after cancellation before permanent deletion. You may reactivate your account within this window without losing data.

Payment Methods
NovaSaaS accepts Visa, Mastercard, American Express, and ACH transfers for annual plans. Invoiced billing is available for Enterprise plans. All payments are processed through Stripe. NovaSaaS does not store raw credit card data.

Invoices and Receipts
Invoices are automatically generated and emailed to the billing contact on each renewal date. Past invoices are available in Settings > Billing > Invoice History. For custom invoicing requirements, contact billing@novasaas.io.

Failed Payments
If a payment fails, NovaSaaS sends an email notification and retries the charge after 3, 7, and 14 days. After three failed attempts, the account is downgraded to the free tier. Service interruption notifications are sent 7 days before downgrade.
            """.strip(),
            metadata={"version": "2.3", "last_updated": "2025-10-01"},
        ),
        SourceDocument(
            doc_id="doc_pricing_001",
            title="Pricing Plans and Feature Tiers",
            source_type="guide",
            category="pricing",
            full_text="""
NovaSaaS Pricing Plans

Starter Plan — Free
The Starter plan is permanently free and supports up to 5 user seats. It includes the core project management features, 10 GB of storage, email support with 48-hour response SLA, and access to the NovaSaaS API at 1,000 requests per minute. The Starter plan does not include SSO, advanced analytics, or custom roles.

Growth Plan — $29/seat/month
The Growth plan supports up to 25 user seats. Additional seats can be purchased at $29 per seat per month. Features include all Starter features plus: 100 GB storage, priority email support with 12-hour response SLA, advanced analytics dashboard, webhook integrations, custom branding, and 10,000 API requests per minute. A 14-day free trial of the Growth plan is available with no credit card required. After 14 days, the account automatically downgrades to Starter unless a paid plan is selected.

Business Plan — $49/seat/month
The Business plan supports up to 50 user seats at $49 per seat per month. Features include all Growth features plus: SSO via SAML 2.0, audit logs, custom roles and permissions, advanced workflow automation, dedicated account manager, and priority phone and chat support with 4-hour response SLA. Business plan customers also receive a 99.5% uptime SLA.

Enterprise Plan — Custom pricing
The Enterprise plan is designed for organizations with more than 50 users or special compliance requirements. Features include all Business features plus: unlimited seats, 99.9% uptime SLA, dedicated infrastructure (optional), custom data residency, HIPAA BAA availability, 24/7 dedicated support with 1-hour response SLA for critical issues, SSO with SCIM provisioning, and custom integrations. Enterprise pricing is negotiated annually. Contact sales@novasaas.io for a quote.

Free Trial
A 14-day free trial of the Growth plan is available with no credit card required. All Growth features are available during the trial. After 14 days, accounts automatically downgrade to Starter.

Nonprofit and Startup Discounts
Registered nonprofits qualify for a 50% discount on any paid plan. Startups under 2 years old with fewer than 10 employees qualify for a 30% discount on the Growth plan. Contact sales@novasaas.io with supporting documentation.
            """.strip(),
            metadata={"version": "3.1", "last_updated": "2025-11-15"},
        ),
        SourceDocument(
            doc_id="doc_security_001",
            title="Security Architecture and Access Control",
            source_type="guide",
            category="security",
            full_text="""
NovaSaaS Security Architecture

Encryption
All data stored in NovaSaaS is encrypted at rest using AES-256. Data in transit is protected using TLS 1.2 or higher. Encryption keys are managed through AWS KMS and rotated annually. Database backups are encrypted separately with a unique key per backup.

Single Sign-On (SSO)
NovaSaaS supports SAML 2.0-based SSO with the following identity providers: Okta, Azure Active Directory (Azure AD), Google Workspace, and OneLogin. SSO is available on the Business and Enterprise plans. To configure SSO, navigate to Settings > Security > Single Sign-On and follow the setup wizard. A test connection must be verified before SSO is enforced. SSO enforcement (blocking password login) is available on the Enterprise plan.

SCIM Provisioning
SCIM (System for Cross-domain Identity Management) provisioning allows automatic user creation, deprovisioning, and group sync from your identity provider. SCIM is available on the Business and Enterprise plans and supports Okta and Azure AD as SCIM providers.

Multi-Factor Authentication (MFA)
MFA is supported via TOTP authenticator apps (Google Authenticator, Authy, 1Password) and hardware security keys (FIDO2/WebAuthn). MFA enforcement across all users can be configured by workspace admins. SMS-based MFA is not supported due to SIM-swap risks.

Role-Based Access Control
NovaSaaS uses a role-based access control (RBAC) model. Built-in roles include Owner, Admin, Member, and Viewer. Custom roles with granular permissions are available on the Business and Enterprise plans. Permissions can be scoped to projects, workspaces, or organization level.

Security Certifications
NovaSaaS holds SOC 2 Type II certification, audited annually by a third-party auditor. The SOC 2 report is available to Enterprise customers upon NDA execution. NovaSaaS is also ISO 27001 certified (since 2024) and undergoes quarterly penetration testing by an independent security firm.

Incident Response
NovaSaaS maintains a 24/7 security operations center. In the event of a data breach, affected customers are notified within 72 hours per GDPR requirements. A full incident report is provided within 30 days. The incident response plan is reviewed and tested biannually.
            """.strip(),
            metadata={"version": "2.0", "last_updated": "2025-09-20"},
        ),
        SourceDocument(
            doc_id="doc_compliance_001",
            title="Compliance, Privacy, and Data Governance",
            source_type="policy",
            category="compliance",
            full_text="""
NovaSaaS Compliance and Privacy

GDPR Compliance
NovaSaaS is fully compliant with the General Data Protection Regulation (GDPR). We act as a data processor for customer data. A Data Processing Agreement (DPA) is available for all customers and is automatically included in the Enterprise contract. For Growth and Business customers, the DPA can be executed via the Trust Center at trust.novasaas.io.

Data Subject Requests
NovaSaaS provides a Data Subject Request (DSR) portal for EU residents to exercise their rights under GDPR, including: right of access, right to rectification, right to erasure (right to be forgotten), right to data portability, and right to restrict processing. Requests submitted through the portal are processed within 30 days. For urgent requests, contact privacy@novasaas.io.

Data Residency
By default, NovaSaaS stores all customer data in the United States (AWS us-east-1). Enterprise customers may request data residency in the European Union (AWS eu-west-1) or United Kingdom (AWS eu-west-2) at no additional charge. Data residency requests must be made during onboarding; migration of existing data incurs a one-time fee.

HIPAA
NovaSaaS supports HIPAA-compliant workflows for healthcare organizations on the Enterprise plan. A Business Associate Agreement (BAA) is available for Enterprise customers. HIPAA-compliant deployment requires enabling specific configuration settings documented in the Enterprise HIPAA guide. NovaSaaS does not claim HIPAA compliance for non-Enterprise plans.

SOC 2 Type II
NovaSaaS's SOC 2 Type II report covers the Security, Availability, and Confidentiality trust service criteria. The report is available to Enterprise customers under NDA. A summary of controls is publicly available at trust.novasaas.io.

Data Retention
Customer data is retained for the duration of the subscription plus 30 days after cancellation (soft-delete period). After 30 days, data is permanently deleted from production systems. Compliance archives (audit logs, access records) are retained for 7 years per regulatory requirements. Customers may request early deletion via the DSR portal.

California Consumer Privacy Act (CCPA)
NovaSaaS complies with CCPA. California residents may submit requests for data access, deletion, or opt-out of sale of personal information at trust.novasaas.io. NovaSaaS does not sell personal information to third parties.
            """.strip(),
            metadata={"version": "1.8", "last_updated": "2025-10-15"},
        ),
        SourceDocument(
            doc_id="doc_onboarding_001",
            title="Onboarding Guide for New Workspaces",
            source_type="guide",
            category="onboarding",
            full_text="""
NovaSaaS Onboarding Guide

Getting Started
After signing up, you will be guided through a 5-step onboarding wizard: (1) Create your workspace, (2) Invite team members, (3) Configure your first project, (4) Set up integrations, and (5) Explore the dashboard. The wizard can be skipped and revisited at any time from Settings > Onboarding.

Inviting Users
To invite users to your workspace, go to Settings > Members > Invite Members. Enter email addresses (comma-separated for bulk invites) and assign a role (Admin, Member, or Viewer). Invited users receive an email with a sign-up link valid for 72 hours. Admins can resend invitations from the Members page.

Data Import and Migration
NovaSaaS supports data migration from CSV, JSON, and Excel (XLSX) formats. The migration wizard is accessible from Settings > Data Import. The wizard guides users through field mapping, data validation, and preview before committing the import. For large datasets (more than 100,000 records), the Support team provides a managed migration service at no additional cost for Business and Enterprise customers. Supported source systems for automated migration include Salesforce, HubSpot, Notion, Airtable, and Monday.com.

Notification Channels
NovaSaaS supports email, Slack, and Microsoft Teams as notification channels for alerts and workflow automations. SMS notifications are available on the Enterprise plan. Webhook endpoints can also be configured as custom notification targets. Notification preferences are managed per-user in Profile > Notifications or globally by admins in Settings > Notifications.

First Project Setup
A project in NovaSaaS represents a unit of work (e.g., a team, a client, a product). Create your first project from the dashboard by clicking '+ New Project'. Configure project settings including access control, custom fields, and default workflow stages. Projects can be made public (visible to all workspace members) or private (invite-only).

Getting Help
In-app help is available by clicking the '?' icon in the sidebar. The NovaSaaS Help Center (help.novasaas.io) contains step-by-step guides, video tutorials, and a searchable knowledge base. For direct support, use the in-app chat widget or email support@novasaas.io.
            """.strip(),
            metadata={"version": "4.0", "last_updated": "2025-12-01"},
        ),
        SourceDocument(
            doc_id="doc_integrations_001",
            title="Integrations and API Reference Overview",
            source_type="guide",
            category="integrations",
            full_text="""
NovaSaaS Integrations

Native Integrations
NovaSaaS offers 50+ native integrations with popular business tools. One-click integration setup is available for: Slack, Microsoft Teams, Google Workspace, Zoom, Salesforce, HubSpot, Zendesk, Jira, GitHub, GitLab, Figma, Dropbox, Box, Google Drive, OneDrive, Stripe, QuickBooks, and many others. Native integrations include automatic field mapping and do not require developer configuration. Integration management is available at Settings > Integrations.

Webhooks
Webhooks allow NovaSaaS to send real-time HTTP POST notifications to any external URL when specific events occur. Webhooks are useful for integrating with custom applications or systems not available in the native integration library. Webhook setup requires: (1) providing the target HTTP endpoint URL, (2) selecting the event types to subscribe to, (3) optionally configuring a secret for signature verification. NovaSaaS signs webhook payloads with an HMAC-SHA256 signature in the X-NovaSaaS-Signature header. Developers should verify this signature to prevent spoofed requests. Error handling and retries: NovaSaaS retries failed webhook deliveries up to 5 times with exponential backoff. Failed deliveries are logged in Settings > Webhooks > Delivery Logs.

Compared to native integrations, webhooks offer greater flexibility and can connect NovaSaaS to any HTTP endpoint. However, they require developer configuration including payload parsing, signature verification, and error handling. For non-technical teams, native integrations are recommended. Webhooks are available on the Growth plan and above.

REST API
NovaSaaS provides a versioned REST API (current version: v2). API authentication is handled via Bearer tokens generated in Settings > Developer > API Tokens. All API responses use JSON. Rate limits: 1,000 requests per minute for Starter, 10,000 requests per minute for Growth and above. When the rate limit is exceeded, the API returns HTTP 429 Too Many Requests with a Retry-After header indicating when the next request may be made. Developers should implement exponential backoff for retry logic. Permanent rate limit increases can be requested through the developer portal at developers.novasaas.io.

GraphQL API
A GraphQL API is available in beta for Enterprise customers. It supports all core resource types and allows clients to request exactly the data they need, reducing over-fetching. Documentation is available at developers.novasaas.io/graphql.

Zapier and Make
NovaSaaS is available as a Zapier app with 200+ supported triggers and actions. A Make (formerly Integromat) module is also available. These low-code integration platforms are recommended for non-developers who need custom automation workflows.
            """.strip(),
            metadata={"version": "2.5", "last_updated": "2025-11-01"},
        ),
        SourceDocument(
            doc_id="doc_api_001",
            title="API Rate Limiting and Developer Guidelines",
            source_type="guide",
            category="api",
            full_text="""
NovaSaaS API Developer Guidelines

API Rate Limits
NovaSaaS enforces rate limits to ensure platform stability and fair usage. Rate limits are applied per API token (not per IP address). Current limits: Starter plan — 1,000 requests per minute; Growth plan — 10,000 requests per minute; Business and Enterprise plans — 10,000 requests per minute (custom limits available on request). When the rate limit is exceeded, the API returns an HTTP 429 Too Many Requests response. The response includes a Retry-After header with the number of seconds until the rate limit resets, and a X-RateLimit-Remaining header with the number of remaining requests in the current window.

Handling Rate Limit Errors
Developers should implement exponential backoff when receiving 429 responses. A recommended approach: on the first 429, wait 1 second; double the wait time on each subsequent 429 (e.g., 1s, 2s, 4s, 8s) up to a maximum wait of 60 seconds. Add jitter (random delay ±10%) to prevent thundering herd. Permanent rate limit increases for legitimate high-volume use cases can be requested through the developer portal at developers.novasaas.io.

API Versioning
The current stable API version is v2. API v1 is deprecated and will be sunset on March 31, 2026. All new integrations should use v2. Breaking changes in v2 will be communicated with at least 90 days notice via email to registered API users and the changelog at developers.novasaas.io/changelog.

Authentication
API tokens are generated in Settings > Developer > API Tokens. Tokens are scoped to specific permissions (read-only, read-write, admin). Best practices: rotate tokens every 90 days, use environment variables (not hardcoded strings) to store tokens, use the minimum required scope. OAuth 2.0 is available for third-party application developers.

Error Codes
HTTP 400 Bad Request: invalid parameters. HTTP 401 Unauthorized: missing or invalid API token. HTTP 403 Forbidden: token lacks required scope. HTTP 404 Not Found: resource does not exist. HTTP 422 Unprocessable Entity: validation error (details in response body). HTTP 429 Too Many Requests: rate limit exceeded. HTTP 500 Internal Server Error: NovaSaaS-side error (report to support).

Pagination
List endpoints return paginated results with a maximum of 100 items per page. Use the cursor parameter for cursor-based pagination (preferred) or page and per_page for offset-based pagination. The response includes next_cursor (or null if last page) and total_count.
            """.strip(),
            metadata={"version": "2.2", "last_updated": "2025-10-20"},
        ),
        SourceDocument(
            doc_id="doc_sla_001",
            title="Service Level Agreement and Support Tiers",
            source_type="policy",
            category="sla",
            full_text="""
NovaSaaS Service Level Agreement (SLA)

Uptime Commitments
NovaSaaS commits to the following uptime SLAs by plan: Starter — best effort (no SLA guarantee); Growth — 99.5% monthly uptime; Business — 99.5% monthly uptime with email + phone support escalation; Enterprise — 99.9% monthly uptime with dedicated support. Uptime is measured as the percentage of minutes in a calendar month during which the NovaSaaS platform is available and responding to requests. Planned maintenance windows (communicated at least 48 hours in advance) are excluded from uptime calculations.

SLA Credits
If uptime falls below the committed level, customers are eligible for service credits: 99.0%–99.5% uptime → 10% credit; 95.0%–99.0% → 25% credit; below 95.0% → 50% credit. Credits are applied to the next invoice. To claim an SLA credit, submit a request within 30 days of the incident via the billing portal.

Support Tiers
Starter: Community forums and Help Center documentation only. No direct support SLA. Growth: Email support with 12-hour first-response SLA during business hours (9 AM–6 PM ET, Monday–Friday). Business: Priority email, phone, and chat support with 4-hour first-response SLA. Dedicated account manager assigned. Enterprise: 24/7 dedicated support channel with 1-hour first-response SLA for P1/critical issues. Dedicated technical account manager and optional quarterly business reviews.

Issue Severity Levels
P1 Critical: complete service outage or data loss risk — Enterprise SLA: 1-hour response, 4-hour resolution target. P2 High: major feature unavailable, significant business impact — Enterprise SLA: 4-hour response, 24-hour resolution target. P3 Medium: partial feature degradation — Enterprise SLA: 8-hour response, 72-hour resolution target. P4 Low: minor issues, questions, feature requests — Enterprise SLA: 24-hour response, next-sprint consideration.

Status Page
Real-time platform status and incident history are available at status.novasaas.io. Subscribe to status updates via email, Slack, or webhook to receive notifications when incidents occur or are resolved.
            """.strip(),
            metadata={"version": "1.5", "last_updated": "2025-08-30"},
        ),
        SourceDocument(
            doc_id="doc_storage_001",
            title="Storage, Files, and Data Limits",
            source_type="guide",
            category="storage",
            full_text="""
NovaSaaS Storage and Data Limits

File Attachments
NovaSaaS supports file attachments in projects, tasks, and messages. The maximum file size per attachment is 50 MB. Supported file types include documents (PDF, DOCX, XLSX, PPTX, TXT), images (PNG, JPG, GIF, WebP, SVG), audio (MP3, WAV), video (MP4, MOV up to 50 MB), and archives (ZIP, TAR.GZ). Files are scanned for malware upon upload. Malicious files are quarantined and the uploader is notified.

Storage Quotas by Plan
Starter plan: 10 GB total storage per workspace. Growth plan: 100 GB total storage per workspace. Business plan: 500 GB total storage per workspace. Enterprise plan: unlimited storage (subject to fair use policy). Storage usage is visible in Settings > Storage. Workspaces approaching 90% of their quota receive email notifications. Exceeding the quota prevents new file uploads until storage is freed or the plan is upgraded.

Data Export
Workspace admins can export all workspace data from Settings > Data > Export. Exports are available in JSON format (complete, machine-readable) or CSV format (flat, for spreadsheet analysis). Export jobs for large workspaces may take up to 24 hours. Completed export files are available for download for 7 days before automatic deletion. API-based data export is available for Enterprise customers for programmatic data portability.

Database Limits
Individual records: maximum 1 MB per record (including all custom fields). Custom fields per project: up to 50 on Growth, 200 on Business and Enterprise. List items per project: up to 10,000 on Growth, unlimited on Business and Enterprise. API response size: up to 10 MB per response.
            """.strip(),
            metadata={"version": "1.2", "last_updated": "2025-07-15"},
        ),
        SourceDocument(
            doc_id="doc_data_policy_001",
            title="Data Lifecycle and Deletion Policy",
            source_type="policy",
            category="data_governance",
            full_text="""
NovaSaaS Data Lifecycle Policy

Active Account Data
All customer data in active accounts is stored indefinitely until explicitly deleted by the customer or until the account is cancelled. Customers retain full ownership of their data. NovaSaaS does not use customer data for training machine learning models or for advertising purposes.

Account Cancellation and Data Deletion
Upon account cancellation, NovaSaaS retains all workspace data for 30 days in a soft-deleted state. During this period, the account can be reactivated and all data fully restored. To request reactivation, contact support within the 30-day window. After 30 days, all data is permanently deleted from production databases, object storage, and CDN caches. Permanent deletion is irreversible.

Compliance Archives
Audit logs, access records, and security event logs are retained for 7 years as required by common regulatory frameworks (SOC 2, GDPR, HIPAA). These records are stored in an isolated, immutable compliance archive with restricted access. They are not available to customers directly but can be provided in response to legal process or regulatory requests.

Right to Erasure (GDPR)
EU residents may request erasure of their personal data via the Data Subject Request portal at trust.novasaas.io. Erasure requests are processed within 30 days. Erasure applies to personal data in production systems. Data retained in compliance archives for regulatory purposes may be retained beyond the erasure request, as permitted under GDPR Article 17(3)(b).

Backup Retention
Automated daily backups are retained for 30 days. Weekly backups are retained for 90 days. Monthly backups are retained for 1 year. Backups are stored in a separate AWS region from production data (cross-region replication). Backup restoration is available on request for Enterprise customers.

Data Portability
Customers may export their data at any time in machine-readable formats (JSON, CSV). See the Storage and Files guide for export instructions. NovaSaaS commits to providing data portability within 72 hours of a request.
            """.strip(),
            metadata={"version": "1.3", "last_updated": "2025-09-05"},
        ),
    ]


def main() -> None:
    setup_logging()
    cfg = load_config()

    print("=" * 60)
    print("NovaSaaS Knowledge Base — Data Preparation")
    print("=" * 60)

    # Build and save corpus
    docs = build_corpus()
    loader = CorpusLoader(cfg.raw_dir(), cfg.processed_dir())
    corpus_path = loader.save_documents(docs)
    print(f"✓ Saved {len(docs)} source documents → {corpus_path}")

    # Print corpus stats
    print(f"\nCorpus summary:")
    for doc in docs:
        print(f"  [{doc.doc_id}] {doc.title} — {doc.word_count} words, category={doc.category}")

    # Chunk corpus
    chunker = Chunker.from_config(cfg._raw)
    chunks = chunker.chunk_corpus(docs)
    chunks_path = loader.save_chunks(chunks)
    print(f"\n✓ Chunked into {len(chunks)} chunks → {chunks_path}")

    # Print chunking stats
    stats = loader.corpus_stats(docs, chunks)
    print(f"\nCorpus statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n✓ Data preparation complete.")


if __name__ == "__main__":
    main()
