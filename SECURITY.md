# Security Policy

## Supported Versions

| Version | Supported           |
| ------- | ------------------- |
| 0.16.x  | Yes                 |
| 0.15.x  | Security fixes only |
| < 0.15  | No                  |

## Reporting a Vulnerability

If you discover a security vulnerability in Selectools, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please email: **support@nichevlabs.com**

You will receive an acknowledgement within 48 hours. We will work with you to understand the issue and coordinate a fix before any public disclosure.

### What to include

- Description of the vulnerability
- Steps to reproduce
- Affected version(s)
- Potential impact

### What to expect

1. **Acknowledgement** within 48 hours
2. **Assessment** and severity classification within 5 business days
3. **Fix and release** — critical issues within 7 days, others within 30 days
4. **Credit** — you will be credited in the release notes (unless you prefer anonymity)

## Built-in Security Features

Selectools includes multiple layers of security for AI agent deployments:

- **Tool Output Screening** — 15 built-in patterns detect prompt injection in tool outputs
- **Coherence Checking** — LLM-based verification that tool calls match user intent
- **Input/Output Guardrails** — PII redaction, topic blocking, toxicity detection
- **Audit Logging** — JSONL trail with privacy controls (redact, hash, omit)
- **Tool Policy Engine** — Declarative allow/review/deny rules with human-in-the-loop

See the [Security documentation](https://johnnichev.github.io/selectools/modules/SECURITY/) for details.
