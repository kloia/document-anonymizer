# Contributing to Document Anonymizer

Thank you for your interest in contributing to Document Anonymizer! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Async-First Rules](#async-first-rules)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment (see below)
4. Create a branch for your changes
5. Make your changes and commit them
6. Push to your fork and submit a pull request

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/document-anonymizer.git
cd document-anonymizer

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Copy environment file
cp .env.example .env

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=document_anonymizer

# Run specific test file
pytest tests/test_dummy_generator.py
```

### Running Linters

```bash
# Run ruff linter
ruff check src/

# Run ruff formatter check
ruff format --check src/

# Run mypy type checker
mypy src/
```

## Code Style

We follow strict code quality standards:

### General Rules

- **PEP 8** compliance with 100 character line limit
- **Type hints** on all function signatures
- **Google-style docstrings** for all public functions
- **Explicit imports** - no `from x import *`
- Use **pathlib** over `os.path`
- Use **f-strings** over `.format()` or `%`

### Example

```python
from pathlib import Path
from typing import Optional

async def process_document(
    file_path: Path,
    output_dir: Path,
    dry_run: bool = False
) -> dict[str, Any]:
    """Process a document for anonymization.

    Args:
        file_path: Path to the input document.
        output_dir: Directory for output files.
        dry_run: If True, analyze without modifying.

    Returns:
        Processing report with statistics.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If file format is unsupported.
    """
    ...
```

### Tools

- **ruff**: Linting and formatting
- **mypy**: Static type checking
- **pre-commit**: Git hooks for automated checks

## Async-First Rules

This project follows async-first architecture. These rules are **mandatory**:

### Required

| Use | For |
|-----|-----|
| `httpx.AsyncClient` | HTTP requests |
| `aiofiles` | File I/O (when needed) |
| `asyncio.gather` | Concurrent operations |
| `asyncio.to_thread` | CPU-bound operations only |

### Forbidden

| Don't Use | Reason |
|-----------|--------|
| `requests` | Blocking HTTP |
| `open()` for I/O | Blocking file operations |
| `ThreadPoolExecutor` for I/O | Use async instead |

### Example

```python
# Good
async with httpx.AsyncClient() as client:
    response = await client.post(url, json=data)

# Bad - DO NOT DO THIS
response = requests.post(url, json=data)  # Blocking!
```

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Code style (formatting, no logic change) |
| `refactor` | Code refactoring |
| `perf` | Performance improvement |
| `test` | Adding/updating tests |
| `chore` | Maintenance tasks |
| `ci` | CI/CD changes |

### Examples

```bash
feat(detector): add support for German ID formats
fix(masker): correct bounding box calculation for rotated text
docs(readme): add troubleshooting section
test(dummy): add format-preserving tests for license plates
```

### Rules

- Use lowercase for type and scope
- Use imperative mood ("add" not "added")
- Keep first line under 72 characters
- Reference issues in footer: `Fixes #123`

## Pull Request Process

### Before Submitting

1. **Update your branch** with the latest main:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Run all checks**:
   ```bash
   ruff check src/
   ruff format --check src/
   mypy src/
   pytest
   ```

3. **Update documentation** if needed

4. **Add tests** for new functionality

### PR Guidelines

- Fill out the PR template completely
- Link related issues
- Keep PRs focused - one feature/fix per PR
- Respond to review feedback promptly

### Review Process

1. Automated checks must pass
2. At least one maintainer approval required
3. All conversations must be resolved
4. Squash merge preferred for clean history

## Reporting Issues

### Bug Reports

Please include:

- Python version and OS
- Document Anonymizer version
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces
- Sample document (if possible, with sensitive data removed)

### Feature Requests

Please include:

- Use case description
- Proposed solution
- Alternatives considered
- Willingness to contribute

### Security Issues

**Do not open public issues for security vulnerabilities.**

See [SECURITY.md](SECURITY.md) for responsible disclosure process.

## Questions?

- Open a [Discussion](https://github.com/kloia/document-anonymizer/discussions) for questions
- Check existing issues before opening new ones
- Join our community discussions

---

Thank you for contributing to Document Anonymizer!
