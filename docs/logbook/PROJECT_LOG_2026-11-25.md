# Project log — 2026-11-25

## Round 2 opened

- The repository now exists on GitHub; all Round 1 commits are pushed and Round 1 is tagged `round-1`.
- Direction from the user: stop patching the synthetic single-difficulty setup, do not finish the wide baseline, and rethink datasets and reward first, with small pilots before any population run. Later the same day: also compare seeding with higher-level epistemic stances against low-level heuristics and plain reasoning, distilled without letting members recite philosophical vocabulary.
- The design is in `docs/design/round-2.md`; the build contract for the three concurrent work packages is in `docs/dev/ROUND2_SPEC.md`; the seeding contexts are in `configs/contexts.yaml`.

## Facts that shaped the design

- **Round 1 wasted most of its rollouts.** Only 16 to 28 of 128 prompts per step had attempts with different rewards; the rest were all-fail or all-succeed and gave no gradient. A census of the base model, eight attempts per item, now sorts training items into sweet spot, frontier and mastered, and the training mix is 70 / 15 / 15.
- **Round 1's loss had no length bias of its own.** It normalised by the token count of the whole batch, not per sequence, so the drift toward long answers was not an artefact of the loss. The flat truncation penalty is replaced by a slope that starts at 2,048 of 4,096 tokens.
- **Standardising advantages is incompatible with a soft slope**: in a group whose attempts are otherwise equal, a tiny length difference would be scaled to full size. Advantages are now centred with a fixed scale.
- **Sampling was never seeded**, and all members of a run share one prompt stream. Clone pairs for the heritability test get explicit, different sampling seeds and the same prompts.
- **Several public math sets leak the sealed questions.** All of MATH-500 is in the MATH test split; GSM-Plus and GSM-Symbolic are rewrites of GSM8K test; DeepScaleR contains MATH-train and most of Omni-MATH. Rewrites are excluded, and texts are de-duplicated across sources before the split, blind first.
- A plain read-only sandbox leaves the token file readable to model-written code; the sandbox hides the repository and home directories.

## Decisions

- **The second vault is extended, not replaced.** Its sealed contents stay, and the blind halves of the new sources join it. The spent halves of the first vault become selection material.
- **A program may stand in for a boxed answer on math problems.** This gives the environment one countable, substantive choice of strategy. It is a design decision the census can still overturn if the base model cannot use the option.
- **Stances and tactics are matched by theme** (falsification, economy, foundations, evidence, structure), so that a difference in how well they survive training reflects the level of abstraction and not the theme.
- **One measure of how much of a context a member carries**, the same for every context: the gain in the teacher's prediction of the member's uninstructed answers when the teacher is given the context, scaled between the base model and the instructed teacher.
- **Self-imitation of a member's own successes** is the amplification variant under test, alongside the relative-only update.
- Qwen3-1.7B stays the base model for the pilots; Qwen3-4B-Instruct is cached and is the fallback teacher.

## In progress

- Three work packages are being built in parallel: verifiers and sandbox; sources, suites and census; engine (soft length slope, centred advantage, self-imitation, distillation seeding, clones, sampling seeds, context score). An independent audit of verifiers, blind separation and the new losses follows before the census is launched.
