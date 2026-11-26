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
- **A program may stand in for a boxed answer on math problems**, to give the environment one countable, substantive choice of strategy. The smoke census overturned this the same day; see below.
- **Stances and tactics are matched by theme** (falsification, economy, foundations, evidence, structure), so that a difference in how well they survive training reflects the level of abstraction and not the theme.
- **One measure of how much of a context a member carries**, the same for every context: the gain in the teacher's prediction of the member's uninstructed answers when the teacher is given the context, scaled between the base model and the instructed teacher.
- **Self-imitation of a member's own successes** is the amplification variant under test, alongside the relative-only update.
- Qwen3-1.7B stays the base model for the pilots; Qwen3-4B-Instruct is cached and is the fallback teacher.

## Build and audits

Four packages were built in parallel (verifiers and sandbox; sources, suites and census; engine; analysis) and three independent audits run on them. Reports are in `docs/dev/audits/`.

- **Smoke census** (100 items per source, base model). Pass rates on training sources range from 0.34 (DeepScaleR) to 0.84 (ARC); responses have a median of about 385 tokens and a 99th percentile of about 2,700, so the length slope starting at 2,048 touches very few; 0.2% are cut off at 4,096.
- **The answer rule was wrong and is changed.** Under "whichever comes last counts", 7% of math samples lost a correct boxed answer to a program appended afterwards that printed nothing (18% on GSM8K). A boxed answer now always wins; a program answers only when there is no box.
- **The program route is dropped.** Once a boxed answer wins, the base model answers by program in 0.2% of math answers, yet the mere offer makes it append an ignored program after the box in 60% of them; forced to use the route it scores 0.23 against 0.40 directly. The offer bought no real choice of strategy and added a large presentational artefact, so math problems now simply ask for a boxed answer. This reverses the morning's decision.
- **Engine audit: all eleven properties hold** (losses, distillation, clone exactness, sampling seeds, resume). One gap, repeated seeds not refused, is closed. The blind guard now also refuses the base model without a committed manifest.
- **Verifier audit, on 18,624 real responses.** One source of false credit: exact values above about 10^40 were compared to 40 digits, so off-by-one answers passed (14 cases, a quarter of the Omni-MATH selection items). Several rejections depended on style: an option letter followed by formatted option text, a correct program followed by a usage example, a program printing a symbolic constant. All fixed; after the fix none of 7,113 perturbed references is accepted. Verdicts also depended on machine load through a wall-clock limit; now only CPU time decides and a starved job is re-run. About 5% of credited code answers are wrong but pass because the dataset's own tests are weak; this is accepted and affects all members alike. No sandbox escape was found.
- **Blind-line audit.** Assignment by item id is sound and Round 1's vault is intact by id. The de-duplication was not: its approximate matching had poor recall and ignored reformatting, leaving an estimated 135 training or selection items with a copy in the vault. The smoke census had sampled the base model on two of them, and the audit also found that three early Round 1 runs hold answers to 57 MMLU-Pro questions that a later re-deal moved into the vault. Both are recorded in `docs/blind-exposures.md`; results at the opening will be given with and without exposed items. The de-duplication was rebuilt (a properly mixing hash; comparison on letters and digits only; a rule that separates copies from problems sharing a template with other numbers). By an independent measure 3 non-blind items still have a copy in the vault, against 135 before. 88 vault items are recorded as exposed, by hash, counting copies and Round 1's evaluations of near-copies in the first vault.
- **Analysis package, validated on simulated truth.** A naive member-by-problem interaction reads pure differences in overall level as divergence (an SD of 2.6 points where none exists); the estimator used compares members within classes of item difficulty and reads zero there. Power at the planned sizes: an interaction SD of about 4 points is detectable with 28 members and 3,000 items; clone pairs sharing half their profile are told from unrelated pairs with six parents; an intraclass correlation needs 14 parents to detect 0.6; the stance-against-tactic comparison on five themes can show an effect size but cannot reach conventional significance.

## Constraints met on the way

- The account's processes share a CPU quota that other jobs of the same account keep saturated. Sandboxed verification is CPU-bound (about 0.3 s per code item), so sandboxes are now reused across jobs and limits are in CPU time.
- An update on 4,096-token rollouts costs about 2.5k tokens per second of trainer time, and with sweet-spot items most rollouts carry gradient. Pilot steps are sized at 48 problems × 4 attempts.

## Census (pilot A)

The base model attempted 27,612 training items eight times each and the censused selection items four times, under the final instructions and verifiers.

- **Pass rates** on training sources: GSM8K 0.86, ARC 0.84, AQuA 0.75, MATH 0.71, ProofWriter 0.69, KodCode 0.65, LogiQA 0.56, MMLU-Pro 0.45, DeepScaleR 0.32; 0.64 overall. On selection units the range is wider, from 0.12 (the Omni-MATH remainder) to 0.92.
- **Format compliance of the base model is 99.6%** on training items and 97.7% on selection items. This is the level members must not lose.
- **Length.** Median 346 tokens, 99th percentile about 3,200; 0.35% cut off at 4,096. DeepScaleR is the long one (median 1,131, a tenth above 2,750, 2.1% cut off), so the slope from 2,048 will bear on hard math and on little else.
- **Contexts move the base model's accuracy little**: 0.44 to 0.54 across the twelve contexts on the same items, with the two most mechanical tactics lowest (`direct` 0.44, `define` 0.47) and three stances and one tactic highest (`parsimony`, `observe`, `refute`, `small_case`, 0.53 to 0.54); the neutral context gives 0.50. A seeded bias is therefore close to reward-neutral at the start.
- The strata table is pinned in run configs by its hash.
- Operationally: the census took about five hours instead of two because every card was shared, and it exposed one more verifier gap. Answers such as a tower of powers made a comparison run forever; these were "undecided" and stopped the run. A comparison now has a CPU budget of 10 s, after which the answer is decided as not verified. One seeding worker that still ran the old code crashed on this at a probe and re-seeded one member.

## Seeding (pilot B), all-domain seeds

Twelve contexts were each distilled into a member in 40 updates of 64 of the member's own answers; a thirteenth member is unseeded. Eleven contexts were distilled a second time on math prompts only, and two (`refute`, `verify`) by Round 1's method, imitation of the teacher's correct instructed answers.

- **Stances are absorbed far more completely than tactics.** On the scale where 0 is the base model and 1 the teacher's own instructed answers, with the drift shared by all contexts removed, the five stances reach 0.25 to 0.34 (mean 0.30) and the six tactics −0.07 to 0.21 (mean 0.07). Two tactics, `map` and `procedure`, are not absorbed at all.
- **In absolute terms the picture is different.** The index is a share of each context's own effect on the teacher, and the stances barely move the teacher: 0.03 to 0.06 nats per token, against 0.09 to 0.31 for the tactics. A stance-seeded member took up 0.010 to 0.015 nats per token. `direct` and `define` took up 0.053 and 0.032, two to four times more, though a smaller share of what was asked; `verify` and `small_case` about as much as a stance; `map` and `procedure`, the contexts that move the teacher most, nothing. So the stances are small, fully general nudges that distil cleanly, and the tactics are large changes of procedure that distil partly or not at all. A member's own context is the top-ranked one for 4 of 11 members (chance 1 in 11).
- **Stances show up in every domain** at similar strength (for `refute`: code 0.39, logic 0.32, math 0.33); tactics are uneven, and four of the six are negative on math.
- **No recitation.** Blacklisted terms appear in 0.3 to 0.8% of responses, the same as for the unseeded member.
- **What is absorbed carries across domains, for stances and tactics alike.** Stances distilled on math alone show up on logic problems at 0.17 to 0.29, against 0.24 to 0.35 when distilled on all domains, so two thirds or more of the full strength; on code the eight probe items are too few to say more than that the sign is the same. The two tactics that were absorbed at all, `direct` and `define`, carry over to logic just as well (0.23 and 0.41).
- **The gap between stances and tactics is a property of the method, not of the level.** Seeded by imitation, the `verify` tactic reaches 0.34 [0.23, 0.48], as high as any stance, against 0.10 [0.04, 0.17] by on-policy distillation; `refute` reaches 0.28 by imitation and 0.33 by distillation. On-policy distillation pulls a member toward the teacher only along answers the member already writes, which suffices for a small general nudge and fails for a change of procedure the member never produces by itself. The claim that survives is narrower: stances are small changes that any method distils cleanly and that appear everywhere at once.
- Accuracy after seeding cannot be read from the single-sample probes; it will come from the evaluation at the end of pilot C.

## Amplification (pilot C), started

Each of the twelve seeds and two unseeded controls was cloned at step 0 into both update rules: 28 members, 45 steps of 48 problems × 4 attempts. In the first steps the training reward is 0.41 to 0.45 and 23 to 29 of 48 problems per step have attempts with different rewards, against 16 to 28 of 128 in Round 1.

An add-on run trains the two imitation seeds under both rules with the same settings, so that the wash-out comparison includes one strongly absorbed tactic.

**Early reading at step 15, from single-sample probes.** The two unseeded members under the relative-only rule are degrading: answers have grown to 1,200 to 1,300 tokens (others 340 to 860), compliance on held-out answer formats is 29 and 44 points below the base weights, and accuracy on mastered items is down 10 and 17 points. No seeded member shows this, the neutral-context one included, and neither do the unseeded members trained with self-imitation. Learning is modest so far: about +4 to +5 points on frontier items in both rules, nothing clear on unseen sources.

## In progress

- Pilot C to step 45, then evaluation, context scores, and the choice of members to clone for the heritability test. The math-only seeds of pilot B.
