You are a senior Python engineer and experienced open-source maintainer.
You are reviewing a project that will be published as a public open-source repository.

Repository name: document-anonymizer

Project purpose:
This is an LLM-based document anonymization system.
It detects sensitive information in documents and replaces them with realistic dummy values.
It also masks visual elements such as signatures and stamps.
The system supports both text-based and vision-based anonymization pipelines.

Repository structure (important):
- src/document_anonymizer/ is the main package
- document_anonymizer.py contains the main orchestrator: DocumentAnonymizer
- cli.py is the CLI entry point (docanon command)
- prompts/ contains LLM prompt templates
- field_detector.py, llm_classifier.py, verification.py handle LLM-based detection & validation
- image_masker.py handles non-reversible visual masking
- dummy_generator.py replaces detected values with dummy data
- OCR, PDF, and rendering logic are separated
- The project is intended for external contributors and production-like usage

Your tasks:

━━━━━━━━━━━━━━━━━━━━━━━━━━
1️⃣ Architecture & Responsibility Review
━━━━━━━━━━━━━━━━━━━━━━━━━━
- Understand the end-to-end flow starting from:
  CLI → DocumentAnonymizer → detection → anonymization → output
- Verify that each module has a single, clear responsibility
- Identify responsibilities that are duplicated or blurred across files
- Check whether orchestration logic leaks into low-level modules

━━━━━━━━━━━━━━━━━━━━━━━━━━
2️⃣ Over-Engineering & Simplification
━━━━━━━━━━━━━━━━━━━━━━━━━━
- Identify over-engineered areas:
  - Unnecessary abstraction layers
  - Over-generalized interfaces
  - Configs or patterns that add complexity without real benefit
- Suggest concrete simplifications
- Do NOT remove correctness, extensibility, or future-proofing where justified

━━━━━━━━━━━━━━━━━━━━━━━━━━
3️⃣ Dead Code & Unused Components
━━━━━━━━━━━━━━━━━━━━━━━━━━
- Find unused or effectively unused:
  - Methods
  - Classes
  - Files
  - Config entries
- Identify code paths that are never executed
- Remove them or clearly mark them for removal

━━━━━━━━━━━━━━━━━━━━━━━━━━
4️⃣ AI-Generated Code Smell Cleanup
━━━━━━━━━━━━━━━━━━━━━━━━━━
This project was partially assisted by AI.

Your job:
- Detect code that *looks obviously AI-generated*, such as:
  - Overly verbose or tutorial-style comments
  - Redundant helper methods
  - Defensive code that never triggers
  - Excessive logging or explanation
  - Unnatural or repetitive naming
- Refactor those parts so the code looks:
  - Human-written
  - Clean
  - Production-grade
  - OSS-maintainer approved

━━━━━━━━━━━━━━━━━━━━━━━━━━
5️⃣ Naming, Structure & Readability
━━━━━━━━━━━━━━━━━━━━━━━━━━
- Improve class, method, and variable naming where needed
- Ensure consistent naming across:
  - detector / classifier / validator terminology
- Suggest folder or file restructuring ONLY if it provides clear value
- Optimize for a new contributor reading the code for the first time

━━━━━━━━━━━━━━━━━━━━━━━━━━
6️⃣ Open Source Readiness (VERY IMPORTANT)
━━━━━━━━━━━━━━━━━━━━━━━━━━
Create a clear checklist covering:

📄 Documentation
- README.md: what must be explained for users and contributors
- Architecture explanation (textual, not diagram generation)
- Example CLI usage and Python API usage
- Limitations and known trade-offs

🔐 Security & Privacy
- Warnings about anonymization guarantees
- What this tool does NOT guarantee (legal compliance, 100% detection)
- Safe handling of prompts and LLM outputs

⚙️ Configuration
- What should be configurable vs hardcoded
- Prompt files exposure considerations
- Environment variables expectations

📦 Repository Hygiene
- What files MUST exist
- What must NEVER be committed (keys, internal prompts, logs)
- Licensing considerations for an AI-based anonymization tool

━━━━━━━━━━━━━━━━━━━━━━━━━━
7️⃣ Output Expectations
━━━━━━━━━━━━━━━━━━━━━━━━━━
- Make direct code changes where improvement is obvious
- For larger architectural suggestions, explain clearly
- Be concise, direct, and critical
- Assume the audience is senior engineers

Constraints:
- Do NOT add new features
- Do NOT rewrite the entire project unless absolutely necessary
- Focus on clarity, correctness, and OSS quality
