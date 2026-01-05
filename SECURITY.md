# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in Document Anonymizer, please report it responsibly.

### How to Report

**Please do NOT open a public GitHub issue for security vulnerabilities.**

Instead, send an email to: **security@kloia.com**

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution Target**: Within 30 days (depending on severity)

### What to Expect

1. We will acknowledge receipt of your report
2. We will investigate and validate the issue
3. We will work on a fix and coordinate disclosure
4. We will credit you in the release notes (unless you prefer anonymity)

### Scope

This policy applies to:
- The Document Anonymizer core library
- Official CLI tools
- Configuration handling
- Data processing pipelines

### Out of Scope

- Third-party dependencies (report to respective maintainers)
- Issues in user-provided configurations
- Denial of service through resource exhaustion with malformed input

## Security Best Practices

When using Document Anonymizer:

1. **Protect your secret key**: Never commit `ANONYMIZATION_SECRET_KEY` to version control
2. **Secure API keys**: Use environment variables for `LLM_API_KEY`
3. **Review outputs**: Always verify anonymized documents before sharing
4. **Keep updated**: Use the latest version for security patches

## Acknowledgments

We thank the security researchers who have helped improve Document Anonymizer through responsible disclosure.
