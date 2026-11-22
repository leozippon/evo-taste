# Project log — 2026-11-22

## Setup

- Repository initialised; `origin` points at `git@github.com:leozippon/evo-taste.git`, which does not exist yet. The available token cannot create repositories, so commits are local until the repository is created.
- Compute: 8× NVIDIA L20 (46 GB), shared with another user and with the owner's other projects. Free GPUs varied between zero and five during the day. All jobs must be resumable and sized to one card.
- Environment: uv-managed, vLLM 0.34.0, torch 2.14, PEFT 0.22, reasoning-gym 0.1.25. Verified recipes are in `docs/dev/INFRA_NOTES.md`.

## Decisions

- **Protocol fixed before any training** (`docs/design/protocol.md`): four hypotheses with refutation criteria, train / selection / blind-vault suites, three seeding arms plus a generic-reasoning control, tolerant culling, harvest on the selection suite only, and a guarded, ledgered blind evaluation with two vaults.
- **Base model: Qwen3-1.7B, thinking disabled.** Reasons: about half the cost of the 4B model per token on this hardware; the literature notes report prompt-induced diversity helping at 1.7B and not at 4B (ICE), and the 4B Instruct model saturating within 20 steps on one paper's procedural task. The 4B calibration on our own tasks is still running; the choice is revisited if it contradicts this.
- **Carrier: LoRA rank 32 on all projections.** Members are adapters on a frozen base; many adapters share one vLLM server per GPU.
- **Environment: reasoning-gym at default configurations, binary exact-match reward.** Suites are assigned by a seeded draw over families, with near-duplicate families kept together and three whole categories held out for selection and the two blind vaults. Families the base model cannot solve at all stay in the training mix on purpose: unlocking one is exactly the kind of rare, self-reinforcing event the project is looking for.
- **External blind benchmarks**: GSM8K test, MATH-500, MMLU-Pro, split between the two vaults; GSM8K train and a third MMLU-Pro split serve as the non-blind foundation probe. `gsm_symbolic` is excluded from the procedural suites so that GSM8K stays out of distribution.

## Observations

- **Qwen3-1.7B is bimodal without thinking.** With reasoning-gym's default prompt, 66% of its answers are a bare `<answer>` with no reasoning (median 17 tokens); its mean score over 105 families at default difficulty is 0.21, counting partial credit, from 60 instances per family. Whether reasoning helps depends on the family: it raises accuracy on `gcd` (0.39 → 0.90), `gsm_symbolic` (0.01 → 0.61) and `base_conversion` (0.00 → 0.79), and lowers it on `number_sequence` (0.84 → 0.47). The base policy therefore already contains a "reason or answer directly" disposition that RL can amplify in either direction. A generic-reasoning control scaffold was added so that style-specific scaffolds are compared against members that reason, not against members that mostly do not.
- **The update, not generation, is the bottleneck.** An L20 trains the 1.7B model at about 4k tokens/s, so one update over 256 rollouts costs roughly 25–30 s, against about 10 s to generate them. Budget consequence: a population of about 30 members for 200 steps is roughly 50–70 GPU-hours, and single-model baselines can be matched to about a quarter of the population's compute within a day on one card each.
- **Literature** (`docs/references/literature-anchors.md`): three of the anchor labels do not match the papers' content (2607.01208 is a bias-auditing paper; 2603.14473 and 2611.02234 concern research-idea taste). The most relevant facts for the design are TAR's result that equally rewarded modes become under-sampled and rarely recover, the primacy-bias paper's finding that RL moves a model far less than supervised fine-tuning does, and the solved-set overlap and preserved/expanded/shrunk accounting used elsewhere as population diagnostics.

## Open

- Engine under construction against `docs/dev/ENGINE_SPEC.md`; an independent audit of the reward, update and isolation paths follows before any pilot.
- Pilot questions: learning rate for amplification; whether seeding leaves a recoverable style; whether any regression appears that would justify targeted protection.
