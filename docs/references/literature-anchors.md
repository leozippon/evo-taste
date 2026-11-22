# Literature anchors for EvoTaste

Notes taken from the arXiv abstract pages and HTML full texts. Where the HTML text was truncated (roughly the last 4-50k characters, mostly appendices), items missing from the main text are marked "not stated". Hyperparameters marked "not stated" were not found in the portion read, so do not assume LoRA or full fine-tuning for them.

| ID | Verified title | Status |
|---|---|---|
| 2611.04061 | On the Primacy Bias in RLVR Training | full text |
| 2507.14843 | The Invisible Leash: Why RLVR May or May Not Escape Its Origin | full text (last ~17k chars unread) |
| 2610.24236 | Preserving Reasoning Diversity in Reinforcement Learning with Verifiable Rewards via Adaptive Replay (method name TAR, Targeted Adaptive Replay) | full text |
| 2611.03627 | Not All Reasoning Generalizes: A Case Study on Learning Generalizable Reasoning Strategies in RLVR | full text (appendix partly unread) |
| 2607.01208 | Distill to Detect: Exposing Stealth Biases in LLMs through Cartridge Distillation (D2D) | full text (appendix unread) |
| 2606.03087 | Learning to Solve, Forgetting to Retain: Correct-Set Turnover in RLVR (method name ReMind) | full text |
| 2602.12275 | On-Policy Context Distillation for Language Models (OPCD) | full text |
| 2608.02087 | Instruction-Conditioned Exploration for Reinforcement Learning with Self-Distillation to an Unconditioned Policy (ICE) | full text |
| 2608.21595 | Perturb the Thought, Not the Pixels: Latent-Space Rollout Diversification for RL of Vision-Language Models (NC-GRPO) | full text |
| 2611.23154 | Learning Stratified Policies for Inference-Time Objectives | full text |
| 2506.10943 | Self-Adapting Language Models (SEAL) | abstract only |
| 2603.14473 | AI Can Learn Scientific Taste (RLCF) | abstract only |
| 2611.02234 | TIGER: Learning from Researcher Taste to Generate Tailored Research Ideas | abstract only |

All IDs resolved. Label caveats: 2607.01208 is a bias-auditing paper, not a general "cartridge distillation for reasoning" paper. 2603.14473 and 2611.02234 are about "scientific taste" in research-idea generation, which shares the word "taste" with EvoTaste but not the meaning. The Invisible Leash is now at v5 (revised 14 Oct 2026); v1 was July 2025.

## 2611.04061 On the Primacy Bias in RLVR Training

Nguyen et al., Nov 2026. Title matches.

- Primacy bias is the tendency to over-rely on early-learned (pre-trained) representations. RLVR from a pre-trained model shrinks Pass@k, while RL from random init does not (MNIST toy with Zipf class skew).
- Training on easy problems mainly sharpens; hard-only training first raises Pass@k and then degrades it as easy problems are forgotten. Extending hard-only training from 512 to 2048 steps lowers overall Pass@k.
- RL stays in a "lazy" regime: KL to the linearised model is 0.04-0.13 for GRPO/DAPO versus 12.7-18.0 for SFT, with update sparsity of roughly 45-80% (the paper itself notes the sparsity contrast is not uniform).
- Pass@128 after 512 steps (OlympiadBench / Minerva / MATH-500): base 68.8 / 53.9 / 94.3; GRPO full 67.9 / 48.8 / 93.2; GRPO hard-only 69.9 / 53.0 / 94.5; GRPO easy-only 66.4 / 49.2 / 92.5.

Mechanism: no new algorithm. GRPO in verl on Qwen2.5-3B-Instruct (Llama-3.2-3B in the appendix), MATH train 7,500 problems, batch 256, 16 rollouts per prompt, constant lr 1e-6, temperature 1.0, max response 3072, no KL or entropy loss, 512 steps. Full-parameter update is implied; single runs, no seeds. Difficulty bins come from pass@1 over 64 samples of the final policy, with zero-accuracy problems discarded.

Metrics: Pass@k coverage shrinkage, update sparsity, NTK drift, and CKA between the initial NTK and the labels. No OOD benchmark is run; the OOD claims rest on prior work.

EvoTaste: the setup for the "different initial biases get amplified" premise. The population's early-learned preferences are the primacy bias to amplify, and the hard/easy mix determines whether members forget their easy skills. A mixed-difficulty environment is a risk for sharpening-only members.

## 2507.14843 The Invisible Leash

Wu et al. (Choi), Jul 2025, v5 Oct 2026. Title matches.

- Introduces "empirical support dynamics": for a verifier R, s_p(x) is the success probability, and the empirical support from n samples is the set of x with at least one verified success. Comparing base q with RLVR model, each question is preserved (both), expanded (only RLVR), shrunk (only base) or out-of-support.
- Metrics: SRR = P/(P+S), NDR = E/(P+E), SDS is their harmonic mean, NSCR = (E-S)/(P+E+S).
- Across six 1.5B-14B comparisons SRR is about 0.93-0.99, NDR is at most 0.02 and NSCR is negative. ProRL-1.5B-v2: P=2388, E=48, S=175, so losses outnumber gains about 3.6:1. AIME24 pass@8192: base 93.3% versus ProRL-v1 83.3%.
- SFT (Qwen2.5-Math-7B, k=256) has NSCR >= 0 on all six benchmarks, while DAPO shows net shrinkage on four. DAPO on DeepSeek-1.5B: Olympiad pass@256 falls from 88.0% to 79.7% by step 300.
- Entropy: ProRL token entropy rises 0.44 to 0.52 while answer entropy falls 1.30 to 0.66; DAPO-32B 0.18 to 0.29 (token) and 1.61 to 0.61 (answer). Token-level entropy therefore does not indicate answer-level diversity.
- Gains are mostly sharpening: Reasoning Gym graph coloring accuracy 96% (ProRL) versus 12% (base) at one attempt, with both reaching 100% at 2048 attempts.

Setup is analysis of released checkpoints (ProRL, Skywork-OR1, AceReason, Nemotron, Phi4-Reason-Plus, OLMo-2-1B, Qwen2.5-Math-7B SFT vs DAPO on DeepMath-103K). Large budgets k of 256-16384, decoding temperature 0.6, top-p 0.95, 32k tokens. The formal content is definitions only, no theorems; training lr, batch size and LoRA/full are not stated.

EvoTaste: copy the support-category accounting (preserved / expanded / shrunk per task) and answer-level entropy as population diagnostics. The per-member "leash" is the expected null result; the population union is what could escape it.

## 2610.24236 TAR (Targeted Adaptive Replay)

Li, Galhotra, Cardie, Oct 2026. The label's description matches the content, although the title is "Preserving Reasoning Diversity ... via Adaptive Replay".

- Theory: the KL-regularised optimum keeps equal-reward modes at their reference ratio, so the objective does not penalise any mode. The skew comes from finite-rollout fluctuations that accumulate; a mode is "under-sampled" when its expected coverage in a group, N times its mass, falls below 1, and such modes rarely recover.
- Toy categorical policy over 300 states with two correct modes: 832 of 900 runs (92.4%) became under-sampled within 1,000 steps, and only 2.6% recovered within the next 200 updates.
- LLM results (Qwen2.5-0.5B/1.5B-Instruct, MATH train): average pass@1 / pass@8 over six benchmarks for 1.5B is base 36.7 / 59.9, GRPO 37.9 / 60.2, TAR 41.2 / 61.1. Diversity at 1.5B (Div-Eq / EAD / Emb-Div): GRPO 0.401 / 0.384 / 0.055; TAR 0.435 / 0.426 / 0.064.

Mechanism: memory bank of correct base-model solutions (N=128 samples per question, BGE-M3 embeddings, furthest-point sampling keeping at most 3 per problem). Replay is gated by discrepancy delta_t = log pi_base - log pi_theta per token. The trajectory gate is g = phi(mean delta) with phi(x) = max(1 - exp(-x), 0), so replay is inactive while the policy is not below base. The loss is J_GRPO minus lambda_rep (=1) times the gate-weighted NLL of the replayed trajectories. Base model is the existing frozen reference. GRPO: 8 rollouts, batch 256, PPO mini-batch 128, AdamW lr 1e-6, clip 0.2, KL 0.001, entropy 0, 3-5 epochs. LoRA/full not stated.

Mode-dominance metric: only in the toy (|log(P1/P2)|, coverage below 1). Real LLM diversity is measured by output metrics (Div-Eq, EAD, Emb-Div, over 8 samples). "Forgotten" means a base solution's sequence probability dropped at least 10x; GRPO shows severe forgetting and TAR about 0-4%.

EvoTaste: the best-grounded mechanism for style divergence. Independent runs with different noise seeds will amplify different modes, and without a replay anchor each run may collapse into one mode. Copy the 10x-probability forgetting check and the toy-level mode statistics as diagnostics.

## 2611.03627 Not All Reasoning Generalizes

Pang and Wang, Nov 2026. Title matches.

- Controlled task: invert non-bijective string functions (given code of f and output y, return k distinct inputs). Reward = min(correct_unique, k) / max(k, m), with 0.9 accuracy and 0.1 format.
- Two strategies: DERIVE (compute digits from constraint equations) and ENUM (guess full assignments and check; the LOOP label is merged into ENUM). They are labelled by a Gemini-3.0-Flash judge (96% agreement with manual labels on 100 traces), and also probed by linear classifiers on hidden states at token offsets 0-1500 after the think tag.
- RLVR-only model: derivation share falls from about 85% in-distribution to below 50% on the three harder shifts, while accuracy stays above 80% on "more answers" and collapses to about 6% on held-out functions with changed lengths.
- Co-training with bijective inversion (1:1) roughly doubles OOD derivation and does much better on the length-change shift. The 1:1 mix gives the best accuracy, because higher bijective ratios cause execution errors. Math zero-shot averages: base 35.1, mono 35.9, mixed 39.8.

Setup: Qwen2.5-7B-Instruct, GRPO with DAPO zero-variance filtering in VERL, 200 steps, 16 prompts x 16 rollouts, lr 1e-6, clip 0.2, entropy 1e-3, no KL, max response 8192, temperature 1.0, 8 H100 for about 1 day per run. Full fine-tuning is probable (FSDP) but not stated. Llama-3.1-8B reached only 27.1% and Qwen3-4B-Instruct saturated at 97.8% in 20 steps. Theory: with terminal reward, GRPO credits each token by its success-probability increment, so strategies with higher "bit rate" win when the answer is unique.

EvoTaste: a template for defining "style" as a labelled strategy (judge plus probe) and for building OOD shifts. The environment mix (here, bijective tasks) decides which strategy survives, so the training mixture is itself a design knob. A task that saturates quickly will show no style divergence.

## 2607.01208 Distill to Detect (Cartridge Distillation)

Talaei et al. (Karbasi, Mirhoseini, Saberi), Jul 2026. Title differs from the label; the paper is about auditing hidden biases, not reasoning.

- D2D trains a cartridge (a prefix-tuning adapter of learned KV states, base weights frozen) so that a frozen base copy matches a suspect model's output distribution. The loss is forward KL from the suspect to the adapted model on 5k unrelated Alpaca prompts.
- The capacity limit keeps the largest coherent part of the logit shift and drops the diffuse residual. Theory: under a diagonal Fisher approximation the optimum is the best rank-k approximation of the shift, and the bias concentration ratio BCR(k) has an inverted-U shape against capacity. Detection peaks at a 16-token cartridge (sweep 4-64).
- Llama-3.2-3B: Petri detection of an owl preference goes from 37% (stealth baseline) to 70% with the cartridge versus 10% (LoRA) and 17% (full); Fanta goes from 33% to 100%, versus 47% for LoRA and full. Qwen3-4B-Instruct: 12% to 40% (owl).
- Biases were implanted by on-policy context distillation, so they live in the logits. They survived GRPO on GSM8K (owl preference 0.94 to 0.92).

Optimizer settings are in the unread appendix.

EvoTaste: a possible diagnostic. A small low-rank or prefix adapter distilled from member minus base could extract the dominant "taste" direction of each member, and the rank-k concentration idea is a way to quantify how coherent a member's shift is. It also confirms that a bias implanted through logits can persist through later RL.

## 2606.03087 ReMind (Correct-Set Turnover)

Qin et al., Jun 2026 (v2 Sep 2026). The title differs from the label; the paper names the method ReMind, and calls its principle "timely-review" rather than repair-window.

- Define q_hat_t(p) from K rollouts and the mastered set M_t as prompts with all K rollouts correct. Retention = |M_t ∩ M_{t+Δ}| / |M_t|, regression = 1 - retention. The graded q_hat also exposes partial regression (e.g. 12/16).
- Timely-review: reviewing a mastered prompt after a delay of 5, 10, 20 or 40 steps costs more retention the longer the delay. 269 of 1,275 review records (about 21%) showed mixed correct and incorrect rollouts.
- Strict non-retention fell from 15.3% to 11.5% (step 200) and 16.4% to 12.3% (step 260) versus GRPO.
- Image-text average over 8 benchmarks: GRPO 60.25 to 64.06 (+3.81), DAPO 61.02 to 64.46, VCRL 63.42 to 64.96. Text math: ExGRPO 51.38 to 52.61.

Mechanism: a FIFO queue of prompt IDs, no stored trajectories. A fresh prompt with q_hat = 1 is enqueued with probability 0.25. Every f = 5 steps, floor(0.10 x batch) review prompts are dequeued and replace part of the fresh batch before rollout (rollout budget unchanged, about 2% of all prompts). Reviewed prompts that stay all-correct graduate; others return to the tail. GRPO-style with clip 0.2 / 0.28, no KL or entropy, K=8, batch 256, lr 1e-6, temperature 1.0, 4096+4096 tokens, about 450 steps, EasyVideoR1. Models: Qwen3-VL-8B-Instruct, Qwen2.5-Math-7B. LoRA/full not stated. Overhead about 3% wall-clock.

EvoTaste: the cheapest retention primitive (prompt IDs only, fixed rollout budget). It can protect each member's mastered set from regressing while its style sharpens. The turnover and retention metrics also work as a per-member diagnostic.

## 2602.12275 OPCD (On-Policy Context Distillation)

Ye et al. (Dong, Wei), Feb 2026. The label matches the paper.

- The student samples y from x alone, and a teacher scores the same y conditioned on [c; x]. The loss is reverse KL(student || teacher) averaged over response tokens, computed on the top-256 student tokens per position. It is on-policy, so it needs no external data or reward.
- Default teacher is a frozen separate copy of the initial weights. Self-distillation (shared weights, simultaneous update) is less stable (Sokoban 18.8 vs 53.9; Medical 50.0 vs 56.8).
- Experiential knowledge: the model writes "experience items" from its own solutions (no labels), and the pool is concatenated into 300 contexts of up to 16k tokens. OPCD then consolidates a chosen context into weights. System-prompt distillation uses optimised MetaSPO prompts.
- Training: 50 steps, batch 128, lr in [1e-6, 5e-6], checkpoints every 2 steps. Models: Qwen3-1.7B/4B/8B, Qwen2.5-3B/7B, Llama-3.2-3B, 3.1-8B. LoRA/full and rollouts per prompt not stated (the algorithm implies one).
- Results: Qwen3-8B math base 75.0, in-context 77.6, off-policy context distillation 78.5, OPCD 79.7 (IFEval 81.3 to 81.7). Qwen3-1.7B Frozen Lake base 6.3, in-context 20.2, OPCD 26.5. Medical (Llama-3.2-3B) base 59.4, context distillation 71.0, OPCD 76.3.
- OOD is measured by IFEval prompt-level strict accuracy (forgetting) and by safety-to-medical transfer.

EvoTaste: the way to turn a style prompt into a prompt-free LoRA initialisation. Seeding a population with distinct initial biases through different context prompts distilled on-policy keeps the base capabilities better than off-policy distillation, but the teacher must be frozen.

## 2608.02087 ICE (Instruction-Conditioned Exploration)

Dilkes, Yazdanpanah, Stein, Aug 2026. The label matches the paper.

- Five fixed strategy instructions are appended to the task prompt: work_backwards, constructive_building, symmetry_and_invariance, decomposition, reduction. They were picked from 10 LLM-proposed candidates by greedy set cover on pass@16 using Qwen3-4B.
- In each group of G=8 rollouts, five carry one instruction each and three use the bare prompt. DAPO trains the instruction-conditioned rollouts (clip 0.2 / 0.28).
- Self-distillation: J = J_T^R - beta_S x KL(sg[teacher on correct rollouts] || student) with beta_S = 0.1. Only correct teacher rollouts are distilled into the unconditioned policy, and the unfiltered forward-KL version is about 12.9% worse than DAPO (single seed), so the correctness filter matters.
- Qwen3-1.7B at 4K tokens: base 0.201, DAPO 0.244, ICE 0.256 (+5.0% relative, 5/5 wins). At 8K, 0.287 to 0.297 (+3.5%). Qwen3-4B at 4K: 0.361 to 0.345 (-4.5%), where the instructions did not expand coverage.
- Pass@256 on AIME/HMMT (1.7B): base 0.512, DAPO 0.483, ICE 0.475, so coverage does not grow.

Setup: OpenRLHF, DAPO-Math-17K (14.1k problems), lr 1e-6, batch 128, T=1.0, 4096 or 8192 tokens, thinking off, about 6,500 GPU-hours. LoRA/full not stated. No explicit diversity metric beyond pass@k.

EvoTaste: a ready primitive for seeding styles (instruction-conditioned exploration) and internalising them. Its negative 4B result shows that instruction diversity can help the 1.7B model and hurt the 4B one, so check the model size before assuming transfer.

## 2608.21595 Latent-Space Rollout Diversification (NC-GRPO)

Jerge, Pelczar, Downes, Aug 2026. Title matches. Vision-language only.

- Half of each rollout group gets Gaussian noise on the final hidden layer during prefill: h_tilde = h + sigma_k x (||h||/sqrt(d)) x eps. This only moves the logits at the last prompt position, so it changes the first generated token. The KV cache is clean and decoding is noise-free.
- Scale is annealed with a sigmoid (gamma=30, k_mid=40), sigma_0 in {0.2, 0.3, 0.4, 0.5}. Group size is 12 (n=6 clean + 6 noisy). The advantage is normalised over the joint group, plus a contrast term (r_clean mean - r_noisy mean)/(2s) with opposite signs per half. Log-probs are computed without noise and without density correction.
- Qwen2.5-VL-7B on Geometry3K (60 steps, batch 128 prompts, 2048 tokens, frozen vision tower): OOD average 52.0 (GRPO) vs 53.0 (sigma 0.2) vs 53.5 (sigma 0.5), with Geo3K in-domain 52.9 to 53.9. Sigma 0.5 regresses on the general battery (p=0.0007). Qwen3-VL-8B-Thinking 60.1 to 61.4; Qwen2.5-VL-3B no change.
- Ablation: antithetic noise pairs drop OOD to 50.1 (below vanilla), so independent per-rollout randomness is the active ingredient, not magnitude or direction.
- Learning rate, KL and LoRA/full are not stated; most configurations are single seed.

EvoTaste: a cheap source of per-member initial-bias variation (noise injected at the first token), and a caution that its gain is modest (about 1-1.5 OOD points) and null at the small 3B scale.

## 2611.23154 Learning Stratified Policies

Mahajan, Zhang, Pacchiano, Nov 2026. Title matches.

- k separate LoRA adapters on one frozen backbone, optimised jointly for pass@k: one response per adapter per prompt, and the objective is the max reward over the k responses. In the linear setting a LoRA population can reach zero best-of-k error even when each member has non-zero error.
- Update rule: GRPO with group-mean baseline, where a single winner is drawn uniformly among correct responses and gets A = r - mean(r), while the others get 0. A leave-one-out variant was worse. No KL (beta = 0).
- Setup: rank 32, k = 5 (also 3 and 8), batch of one prompt per step, AdamW 8-bit, constant lr 1e-6 (5e-6 for Knights-and-Knaves), temperature 1.0, 768 tokens. Models: TinyGPT (arithmetic), Qwen2.5-1.5B-Instruct (Knights-and-Knaves, Sudoku), Llama-3.1-8B-Instruct (GSM-Symbolic). 2,000-10,000 RL steps, usually after a warm start.
- Pass@5: Sudoku 74.2 to 80.3, arithmetic 95.5 to 99.7, Knights-and-Knaves 90.6 to 93.2, GSM-Symbolic 85.5 to 90.4. A 5x-rank single LoRA stays at 74.2 on Sudoku, and a random mixture of the same adapters is worse than Stratified everywhere. Pass@1 falls (Sudoku 40.9 to 34.6).
- Diversity: pairwise Jaccard similarity of per-adapter solved sets (greedy), the count of adapters solving each problem, unique-only solutions per adapter. On Sudoku only 6.0% of problems are solved by all five and 44.0% by exactly one.

EvoTaste: the contrast. Stratified policies train members jointly with a shared reward coupling, whereas EvoTaste trains them independently, so the specialisation here is by design and the mixture control shows that a population only pays off if members actually differ. Reuse Jaccard overlap and solved-by-count as the population diversity metrics.

## 2506.10943 SEAL (abstract only)

Zweiger et al. (Kim, Agrawal), Jun 2025 (v2 Sep 2025). The model generates its own fine-tuning data and update directives ("self-edits"), applied by supervised fine-tuning, and an RL loop uses the updated model's downstream performance as the reward to learn to write better self-edits. Evaluated on knowledge incorporation and few-shot generalisation. No numbers were read.

EvoTaste: contrast only. SEAL learns a weight-update policy through an outer loop, whereas EvoTaste has no outer loop and relies on population selection afterwards.

## 2603.14473 RLCF, "AI Can Learn Scientific Taste" (abstract only)

Tong et al. (Qiu), Mar 2026 (v3 Aug 2026). Reinforcement Learning from Community Feedback trains a "Scientific Judge" from signals such as citations and a "Scientific Thinker" that proposes higher-impact research ideas. The Judge reportedly outperforms strong LLM baselines and generalises to future papers, other community metrics and unseen fields. "Taste" here means judging research impact from community signals; it is not a reasoning style.

## 2611.02234 TIGER (abstract only)

Gu et al. (Chenhao Tan), Nov 2026. TIGER-8B is trained on individual researchers' preferences to generate tailored research ideas. It beat GPT-4.1 in pairwise judgments by three LLM researchers (56.6%-68.4% of 76 comparisons each) and two PhD researchers (63.3% and 56.7% of 30), and each judge preferred the policy matched to them over an off-target policy. Taste here is learned from a preference signal on idea quality, not emergent from RLVR; it is a naming contrast only.
