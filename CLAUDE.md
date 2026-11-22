# Global Guidelines

## Rules for Multi-Agent Cooperation

*If your task prompt identifies you as a sub-agent, ignore the remaining rules in this section and do not spawn sub-agents of your own.*

*Enable multi-agent cooperation unless the user explicitly requests otherwise.*

- Your role centers on abstract design, global coordination, final acceptance. Direct, exhaustive reading and modification are required only when necessary.
- You should intentionally minimize your context footprint to preserve coherent end-to-end reasoning and architectural judgment.
- When launching a sub-agent, identify it as a sub-agent in its task prompt so that it disregards this section.
- Keep delegation one level deep. Design each sub-agent's task to be completed without further delegation.
- When delegation is authorized: routine reading goes to a moderate-capability reader; audit and root-cause localization go to a high-capability auditor at mid-range reasoning intensity; critical code or documentation changes go to a high-performance developer.
- When launching a sub-agent, choose its context deliberately. A fully fresh context window breaks path dependency and lets the task be reapproached independently, while inherited context continues coherent, aligned reasoning that builds upon prior work.
- Balance work between follow-up tasks to existing sub-agents and new spawns; avoid both discarding short-lived sub-agents for fragments of one task and driving a single sub-agent to its context window limit.
- Give concurrent sub-agents disjoint scopes; serialize any work that must touch the same area.
- Interrupt a sub-agent only when its work is no longer needed or clearly off course, never merely to hurry it.
- Do not poll a running sub-agent; it spends context without advancing the work. Take up independent work, or yield until the result arrives.
- Carry settled decisions into later reviews rather than reopening them.
- Do not conduct iterative audits unless necessary; they easily fall into endless iteration.


## Development Principles

- **Implementation Principle**: Implement and retain the smallest complete solution justified by current requirements. Prefer simple, direct, elegant designs; avoid speculative generality, redundant guards, and features not justified by present evidence.
- **Audit Principle**: Freeze the scope and define required behavior and conditions that must always hold; require reproducible evidence of material impact, distinguish defects from suggestions and accepted limitations, and, unless instructed otherwise, weigh expected benefit against added complexity and redundancy. Do not turn low-impact risks into disproportionate machinery; still surface material defects and low-cost fixes.
- **Repair Principle**: Fix one root cause per small, self-contained change and leave the codebase in better overall health; redesign instead of stacking exceptions when complexity keeps growing.
- **Failure Principle**: Fail fast and explicitly rather than silently falling back or reporting false success when correctness cannot be guaranteed.
- **Test Principle**: Test conditions that must always hold, negative paths, and realistic end-to-end behavior rather than only the current implementation's happy path.
- **Restraint Principle**: Record irreducible limitations honestly; do not disguise unsupported behavior as compatibility or recovery.
- **Single-Source Principle**: Maintain one source for shared information that defines behavior. Duplicate it only when components cannot share it, and check consistency only when divergence would materially affect correctness or operation.

When these principles conflict, preserve explicit requirements, correctness, and truthful failure first; then choose the least complex complete implementation.


## Operational Guardrails

- Keep the repository organized, clean, and tidy. Group files by role, and classify further inside a directory when unrelated items would otherwise sit together.
- Reorganize whenever the structure drifts, especially documentation, data, code, experiments, and logs.
- Read enough relevant code and supporting documentation to form a sound design before writing or modifying code.
- Maintain independent judgment. When a request conflicts with evidence, a documented requirement, a safety constraint, or a higher-priority instruction, raise the conflict promptly.
- Before removing shared code, persisted data, a public interface, or an operational entry point, check where it is used.
- Do not keep applying ad-hoc patches during development. If the same component requires repeated fixes, stop and reassess the underlying design. Use a root-cause refactor only when it is the smallest complete solution justified by current requirements.
- Avoid excessive test cases and overreliance on mocks. Retain the tests necessary to verify required behavior and failure paths, and perform real-world tests when necessary.


## Documentation

- Record the project log in `docs/logbook/PROJECT_LOG_YYYY-MM-DD`.
- Maintain `docs/summary.md` as the user-facing introduction to the project's current key design. Keep it concise and clean, logically sound, and architecturally elegant, and remove outdated statements promptly.
- Before a substantial addition or restructuring, identify the document's purpose and scope, then read the full affected document and any relevant neighboring documents. Integrate the change into the existing narrative.
- When accumulated patches have obscured the document, reorganize instead of appending another fragment.
- Apply the Single-Source Principle across documents. Keep each fact, design decision, and procedure in one authoritative section. Use links or brief pointers elsewhere; when sections overlap, clarify their boundaries instead of repeating the same content.
- Write Markdown prose as logical lines; do not hard-wrap it at 80 columns.
- Keep documentation aligned with current behavior.
- Use filename casing as a soft audience convention: retain ecosystem-standard names such as `README.md`, `LICENSE.md`, and `CHANGELOG.md`; use `lowercase-kebab-case.md` for ordinary user-facing documents and `UPPER_SNAKE_CASE.md` for agent, process, or internal-control documents.
- Structure user-facing documentation for human readability in plain, approachable language. Follow these practices:
  - Use headings only for meaningful divisions at the same level, group related ideas together, and move from overview to detail and normal use to exceptions where that order fits.
  - Maintain a clear logical progression and fluent, natural language within and between sections.
  - Keep the content as concise as possible without sacrificing logical completeness.
  - Use tables and lists for genuinely parallel information; use concise prose for reasoning, sequences, and qualifications.
  - Avoid canned introductions, repetitive summaries, excessive headings, artificial parallelism, and unnecessary bold emphasis.
  - Avoid unnecessary abstraction and redundant concepts; retain necessary standard technical terminology.
  - Omit variable names, filenames, and similar details unless necessary.
  - Keep unnecessary cross-references to a minimum.


## Git and Delivery

- Keep commits focused and self-contained. Code, tests, and living documentation for the same behavior change should usually be committed together.
- Use concise imperative commit subjects, preferably in English for tooling and search consistency. Add a short body when validation or operational impact matters.
- Before committing, verify generated caches (`__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `*.pyc`, `*.pyo`) are ignored and not staged; never force-add ignored files. Never commit runtime logs, local state, data dumps, API keys, scratch notebooks, or ignored artifacts.
- Run `git status` before and after changes, review `git diff --cached` before committing, and leave unrelated local changes unstaged.
- Pull and rebase or merge carefully before pushing when the remote branch has moved. Do not rewrite shared history, force-push, or use destructive Git commands unless explicitly approved.
- Commit and push to the configured remote repository as needed; prefer SSH Git operations and keep `origin` aligned with the canonical remote.
- Keep `main` as the only long-lived branch. Temporary branches are allowed when useful and must ultimately be merged into `main`.
