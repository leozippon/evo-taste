# reasoning-gym calibration

Base-model calibration of every reasoning-gym dataset at its default config: 60 instances per dataset (seed 1234), one sample each at temperature 1.0 (top_p 1.0), max_tokens 1024, no thinking phase, reasoning-gym's `default` system prompt, answer taken from the last `<answer>…</answer>` and scored with the dataset's own `score_answer`. Models: Qwen3-1.7B (`enable_thinking=False`) and Qwen3-4B-Instruct-2507. Raw per-instance records are on the compute host under `results/calibration/` (`<model>.jsonl`, `catalogue.json`); regenerate this file with `scripts/dev/rg_calibrate.py`.

Columns: `acc` is mean score; `=1` the fraction scoring exactly 1.0; `no-ans` the fraction without an extractable answer; `tok mean/p95` completion tokens (1024 means truncated). Scoring is classified from the scores of model outputs and of perturbed gold answers: `exact` gives only 0 or 1, `exact+ε` also gives a small format credit (at most 0.05) to wrong but parseable answers, `partial` gives larger fractional credit; `*` marks datasets that use the base-class scorer, which gives `len(gold)/len(answer)` credit whenever the gold answer is a substring of the answer. Difficulty knobs are the dataset's curriculum attributes with their default-config values (`lo–hi` for ranges).

Over all 60 × 105 prompts:

- 1.7b: mean score 0.21, exactly 1.0 0.17, no answer 0.07, truncated at 1024 tokens 0.03; among completed samples: mean score 0.22, no answer 0.04.
- 4b: mean score 0.38, exactly 1.0 0.37, no answer 0.51, truncated at 1024 tokens 0.51; among completed samples: mean score 0.77, no answer 0.01.

Almost every missing answer is a truncated sample: Qwen3-4B-Instruct-2507 writes long step-by-step solutions, so at max_tokens 1024 half of its samples never reach `<answer>`; its per-dataset accuracy here is a lower bound, and its RL runs need a larger token budget or a prompt that asks for brevity.

| dataset | category | scoring | difficulty knobs (default) | 1.7b acc | 1.7b =1 | 1.7b no-ans | 1.7b tok mean/p95 | 4b acc | 4b =1 | 4b no-ans | 4b tok mean/p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_arithmetic | algebra | partial | min_real=-10, max_real=10, min_imag=-10, max_imag=10, operations_weights=[4 values] | 0.70 | 0.65 | 0.00 | 55/509 | 1.00 | 1.00 | 0.00 | 175/515 |
| intermediate_integration | algebra | exact | problem_type_weights=[8 values] | 0.05 | 0.05 | 0.00 | 22/34 | 0.80 | 0.80 | 0.00 | 386/708 |
| polynomial_equations | algebra | partial | degree=1–3, terms=2–4 | 0.41 | 0.17 | 0.00 | 16/29 | 0.78 | 0.60 | 0.22 | 447/1024 |
| polynomial_multiplication | algebra | exact | min_terms=2, max_terms=4, min_value=1, max_value=100, min_degree=0, max_degree=3, min_polynomials=2, max_polynomials=3 | 0.05 | 0.05 | 0.05 | 92/212 | 0.53 | 0.53 | 0.20 | 670/1024 |
| simple_equations | algebra | partial* | min_terms=2, max_terms=4, min_value=1, max_value=100, operators_weights=0.4/0.4/0.2 | 0.80 | 0.78 | 0.17 | 141/192 | 0.99 | 0.98 | 0.00 | 156/232 |
| simple_integration | algebra | exact | terms=2–5 | 0.13 | 0.13 | 0.00 | 29/64 | 0.77 | 0.77 | 0.02 | 322/525 |
| ab | algorithmic | exact | length=10 | 0.00 | 0.00 | 0.00 | 17/27 | 0.00 | 0.00 | 1.00 | 1024/1024 |
| base_conversion | algorithmic | partial* | base=2–16, value=0–1000 | 0.32 | 0.32 | 0.02 | 140/385 | 0.98 | 0.98 | 0.00 | 323/529 |
| binary_alternation | algorithmic | partial* | n=10–30 | 0.18 | 0.18 | 0.00 | 8/8 | 0.02 | 0.02 | 0.98 | 1018/1024 |
| binary_matrix | algorithmic | exact | p_zero=0.25, n=3–10 | 0.00 | 0.00 | 0.00 | 70/170 | 0.08 | 0.08 | 0.92 | 990/1024 |
| caesar_cipher | algorithmic | partial* | rotation=1–25, words=3–20 | 0.00 | 0.00 | 0.00 | 18/22 | 0.00 | 0.00 | 1.00 | 1024/1024 |
| count_primes | algorithmic | partial* | n=1–10000 | 0.00 | 0.00 | 0.05 | 176/288 | 0.00 | 0.00 | 0.13 | 474/1024 |
| cryptarithm | algorithmic | exact+ε | words=2–3 | 0.00 | 0.00 | 0.83 | 951/1024 | 0.00 | 0.00 | 1.00 | 1024/1024 |
| game_of_life | algorithmic | partial | grid_size_x=10, grid_size_y=10, filled_cells_weights=0.1, simulation_steps=1 | 0.87 | 0.00 | 0.00 | 266/319 | 0.00 | 0.00 | 1.00 | 1024/1024 |
| game_of_life_halting | algorithmic | exact | grid_size_x=12, grid_size_y=12, difficulty=1, num_oscillators=5, max_simulation_steps=20 | 1.00 | 1.00 | 0.00 | 8/8 | 0.02 | 0.02 | 0.98 | 1023/1024 |
| graph_color | algorithmic | exact+ε | num_vertices=10–10, num_colors=3 | 0.26 | 0.25 | 0.00 | 65/65 | 0.62 | 0.62 | 0.35 | 800/1024 |
| group_anagrams | algorithmic | exact+ε | anagram_groups=2–10, words_per_group=2–5 | 0.04 | 0.03 | 0.00 | 69/114 | 0.32 | 0.32 | 0.67 | 904/1024 |
| isomorphic_strings | algorithmic | partial* | string_length=2–10 | 0.38 | 0.38 | 0.00 | 23/21 | 0.80 | 0.80 | 0.12 | 460/1024 |
| jugs | algorithmic | exact+ε | num_jugs=3, difficulty=10 | 0.00 | 0.00 | 0.03 | 86/100 | 0.00 | 0.00 | 0.53 | 620/1024 |
| letter_counting | algorithmic | partial* | words=5–15 | 0.18 | 0.18 | 0.00 | 14/8 | 0.93 | 0.93 | 0.00 | 340/533 |
| letter_jumble | algorithmic | partial | word_len=1–64, words=3–20, corruption_level=0.1–0.9 | 0.06 | 0.00 | 0.03 | 62/49 | 0.14 | 0.12 | 0.85 | 960/1024 |
| manipulate_matrix | algorithmic | partial | rows=2–10, cols=2–10, num_transforms=1–10 | 0.04 | 0.03 | 0.02 | 31/62 | 0.30 | 0.30 | 0.63 | 884/1024 |
| number_filtering | algorithmic | partial* | numbers=3–10, decimals=0–4, min_value=-100.0, max_value=100.0 | 0.07 | 0.05 | 0.00 | 31/54 | 0.90 | 0.90 | 0.00 | 356/517 |
| number_sorting | algorithmic | exact | numbers=3–10, decimals=0–2, min_value=-100.0, max_value=100.0 | 0.00 | 0.00 | 0.00 | 41/66 | 1.00 | 1.00 | 0.00 | 263/379 |
| palindrome_generation | algorithmic | exact+ε | length=3–10 | 0.05 | 0.03 | 0.00 | 11/13 | 0.72 | 0.72 | 0.20 | 523/1024 |
| palindrome_partitioning | algorithmic | exact | string_len=5–15, substring_palindrome_len=1–5 | 0.00 | 0.00 | 0.20 | 257/1024 | 0.00 | 0.00 | 0.87 | 987/1024 |
| pool_matrix | algorithmic | partial | rows=2–10, cols=2–10, pool_size=1–3 | 0.10 | 0.10 | 0.02 | 96/358 | 0.39 | 0.38 | 0.52 | 740/1024 |
| ransom_note | algorithmic | partial* | note_length=1–10, magazine_length=2–30 | 0.50 | 0.50 | 0.00 | 8/8 | 0.97 | 0.97 | 0.00 | 413/646 |
| rotate_matrix | algorithmic | partial* | n=2–10, num_rotations=0–10 | 0.15 | 0.15 | 0.02 | 61/199 | 0.52 | 0.52 | 0.45 | 661/1024 |
| rotten_oranges | algorithmic | partial* | n=10–30 | 0.07 | 0.07 | 0.00 | 9/9 | 0.00 | 0.00 | 1.00 | 1024/1024 |
| sentence_reordering | algorithmic | partial | words_in_sentence=3–20 | 0.08 | 0.00 | 0.00 | 19/28 | 0.16 | 0.07 | 0.48 | 727/1024 |
| spell_backward | algorithmic | partial | word_len=3–10 | 0.30 | 0.00 | 0.02 | 10/11 | 0.84 | 0.72 | 0.07 | 211/1024 |
| spiral_matrix | algorithmic | exact | n=2–10 | 0.05 | 0.05 | 0.03 | 115/208 | 0.10 | 0.10 | 0.90 | 968/1024 |
| string_insertion | algorithmic | exact | string_length=5–20 | 0.50 | 0.50 | 0.00 | 13/17 | 0.23 | 0.23 | 0.77 | 940/1024 |
| string_manipulation | algorithmic | partial* | string_length=5–20, num_rules=3–8 | 0.23 | 0.17 | 0.00 | 12/15 | 0.22 | 0.22 | 0.78 | 913/1024 |
| string_splitting | algorithmic | partial* | initial_machines=0–5 | 0.00 | 0.00 | 0.00 | 18/20 | 0.22 | 0.22 | 0.75 | 952/1024 |
| string_synthesis | algorithmic | partial* | initial_blocks=0–5 | 0.12 | 0.12 | 0.00 | 24/24 | 0.30 | 0.30 | 0.70 | 933/1024 |
| word_ladder | algorithmic | partial | word_length=4–4 | 0.00 | 0.00 | 0.02 | 32/36 | 0.05 | 0.05 | 0.73 | 917/1024 |
| word_sequence_reversal | algorithmic | partial* | words=3–8 | 0.20 | 0.20 | 0.00 | 17/23 | 0.63 | 0.63 | 0.00 | 18/23 |
| word_sorting | algorithmic | partial | num_words=3–10, word_length=3–12 | 0.26 | 0.03 | 0.00 | 21/30 | 0.78 | 0.72 | 0.17 | 544/1024 |
| arc_1d | arc | partial* | size=10–30 | 0.07 | 0.07 | 0.35 | 638/1024 | 0.00 | 0.00 | 1.00 | 1024/1024 |
| arc_agi | arc | exact+ε | rotations_weights=[4 values], mirrors_weights=[5 values] | 0.04 | 0.00 | 0.22 | 329/1024 | 0.00 | 0.00 | 0.98 | 944/1024 |
| rearc | arc | exact+ε | pso_difficulty_weights=[7 values], rng_difficulty_weights=[7 values] | 0.19 | 0.15 | 0.03 | 130/427 | 0.07 | 0.07 | 0.93 | 994/1024 |
| basic_arithmetic | arithmetic | partial* | num_terms=2–6, num_digits=1–4 | 0.40 | 0.40 | 0.08 | 110/276 | 0.90 | 0.90 | 0.05 | 261/878 |
| bitwise_arithmetic | arithmetic | exact | difficulty=2 | 0.03 | 0.03 | 0.00 | 91/526 | 0.23 | 0.23 | 0.52 | 794/1024 |
| calendar_arithmetic | arithmetic | partial | tasks=[7 values], offset_upper_bound=100 | 0.37 | 0.25 | 0.03 | 120/522 | 0.65 | 0.60 | 0.18 | 413/1024 |
| chain_sum | arithmetic | partial | num_terms=2–6, num_digits=1–4 | 0.43 | 0.43 | 0.03 | 28/175 | 1.00 | 1.00 | 0.00 | 80/175 |
| count_bits | arithmetic | partial* | n=1–2147483647 | 0.04 | 0.03 | 0.02 | 247/869 | 0.02 | 0.02 | 0.93 | 1007/1024 |
| decimal_arithmetic | arithmetic | exact | decimal_places=3–3, precision=12, num_terms=2–6 | 0.15 | 0.15 | 0.02 | 53/227 | 0.32 | 0.32 | 0.17 | 313/1024 |
| decimal_chain_sum | arithmetic | exact | num_terms=2–6, num_digits=1–4, decimal_places=1–4 | 0.22 | 0.22 | 0.00 | 18/41 | 0.95 | 0.95 | 0.00 | 139/245 |
| dice | arithmetic | exact | num_dice=4, max_dice_size=20 | 0.00 | 0.00 | 0.07 | 378/796 | 0.00 | 0.00 | 1.00 | 1024/1024 |
| fraction_simplification | arithmetic | partial | value=1–1000, factor=1–100 | 0.31 | 0.28 | 0.20 | 423/752 | 0.87 | 0.87 | 0.05 | 560/1023 |
| gcd | arithmetic | partial* | num_terms=2–2, value=1–1000 | 0.81 | 0.77 | 0.00 | 232/410 | 1.00 | 1.00 | 0.00 | 347/573 |
| gsm_symbolic | arithmetic | exact+ε | (no curriculum) difficulty=1.0 | 0.19 | 0.18 | 0.23 | 34/122 | 0.92 | 0.92 | 0.02 | 182/336 |
| lcm | arithmetic | partial* | numbers=2–2, value=1–100 | 0.83 | 0.83 | 0.02 | 149/219 | 1.00 | 1.00 | 0.00 | 195/323 |
| leg_counting | arithmetic | partial* | num_animals=3–10, num_instances=1–15 | 0.18 | 0.18 | 0.00 | 170/363 | 0.72 | 0.72 | 0.02 | 341/602 |
| number_format | arithmetic | exact | num_candidates=2–5, n=1000–1000000000, max_delta=10.0 | 0.63 | 0.63 | 0.00 | 21/25 | 0.65 | 0.65 | 0.07 | 63/347 |
| power_function | arithmetic | exact+ε | exponent=0–8 | 0.06 | 0.05 | 0.00 | 18/28 | 0.47 | 0.47 | 0.07 | 440/1024 |
| prime_factorization | arithmetic | partial | value=2–1000 | 0.50 | 0.45 | 0.02 | 35/184 | 1.00 | 1.00 | 0.00 | 284/483 |
| products | arithmetic | partial | num_terms=2–2, num_digits=1–5 | 0.42 | 0.42 | 0.00 | 13/18 | 0.53 | 0.53 | 0.00 | 13/17 |
| time_intervals | arithmetic | partial | max_time_difference_seconds=86400, max_date_difference_days=100 | 0.56 | 0.28 | 0.00 | 60/377 | 0.60 | 0.52 | 0.40 | 615/1024 |
| bf | code | partial* | difficulty=1 | 0.00 | 0.00 | 0.12 | 133/1024 | 0.00 | 0.00 | 1.00 | 1024/1024 |
| codeio | code | partial | difficulty=None | 0.04 | 0.02 | 0.02 | 36/93 | 0.12 | 0.10 | 0.48 | 653/1024 |
| color_cube_rotation | cognition | exact+ε | rotations=1–3 | 0.17 | 0.17 | 0.00 | 17/8 | 0.08 | 0.08 | 0.88 | 967/1024 |
| figlet_font | cognition | partial | word_len=3–7 | 0.04 | 0.00 | 0.00 | 10/9 | 0.01 | 0.00 | 0.72 | 872/1024 |
| modulo_grid | cognition | partial* | size_x=20, size_y=20, max_holes=1, max_divisor=20, max_target=20 | 0.00 | 0.00 | 0.05 | 237/468 | 0.00 | 0.00 | 1.00 | 1024/1024 |
| needle_haystack | cognition | exact | num_statements=10–100 | 0.63 | 0.63 | 0.02 | 9/11 | 0.82 | 0.82 | 0.15 | 9/14 |
| number_sequence | cognition | partial* | terms=4–8, min_value=-100, max_value=100, max_complexity=3 | 0.57 | 0.57 | 0.00 | 141/436 | 0.66 | 0.65 | 0.22 | 477/1024 |
| rectangle_count | cognition | partial | max_rectangles=10 | 0.10 | 0.00 | 0.00 | 10/11 | 0.14 | 0.13 | 0.78 | 889/1024 |
| rubiks_cube | cognition | exact+ε | cube_size=3, scramble_steps=3–10 | 0.01 | 0.00 | 0.38 | 402/1024 | 0.01 | 0.00 | 0.08 | 366/1024 |
| boxnet | games | partial | row_num=1–4, column_num=2–4, box_num=1–1 | 0.17 | 0.00 | 0.07 | 228/428 | 0.26 | 0.08 | 0.47 | 565/1024 |
| countdown | games | exact+ε | numbers=4–6, target=100–999, value=1–100 | 0.11 | 0.07 | 0.02 | 40/29 | 0.47 | 0.47 | 0.53 | 770/1024 |
| emoji_mystery | games | exact+ε | num_words_in_sentence=3–35 | 0.01 | 0.00 | 0.00 | 31/20 | 0.00 | 0.00 | 1.00 | 1024/1024 |
| futoshiki | games | partial | board_size=4–9, difficulty=0–3 | 0.14 | 0.00 | 0.12 | 182/1024 | 0.00 | 0.00 | 0.97 | 999/1024 |
| kakurasu | games | exact | rows=4–5, cols=4–5, p_ones=0.3 | 0.00 | 0.00 | 0.00 | 13/38 | 0.00 | 0.00 | 1.00 | 1024/1024 |
| knight_swap | games | exact | nodes=6–9, pieces=2–2, steps=4–20 | 0.17 | 0.17 | 0.42 | 18/43 | 0.00 | 0.00 | 1.00 | 1024/1024 |
| mahjong_puzzle | games | partial* | num_rounds=10–50 | 0.20 | 0.20 | 0.07 | 76/1024 | 0.00 | 0.00 | 1.00 | 1024/1024 |
| maze | games | partial* | dist=5–10, grid_size=5–10 | 0.04 | 0.03 | 0.00 | 9/9 | 0.03 | 0.03 | 0.97 | 1016/1024 |
| mini_sudoku | games | partial | empty=8–12 | 0.20 | 0.05 | 0.15 | 151/1024 | 0.00 | 0.00 | 1.00 | 1024/1024 |
| n_queens | games | exact | n=8, num_removed=1–7 | 0.00 | 0.00 | 0.43 | 455/1024 | 0.08 | 0.08 | 0.88 | 1011/1024 |
| puzzle24 | games | exact+ε | value=1–10 | 0.09 | 0.08 | 0.00 | 18/19 | 0.62 | 0.62 | 0.37 | 650/1024 |
| rush_hour | games | exact+ε | moves=1–50 | 0.01 | 0.00 | 0.00 | 17/22 | 0.00 | 0.00 | 1.00 | 1024/1024 |
| sokoban | games | exact | width=6–10, height=6–10 | 0.00 | 0.00 | 0.03 | 46/15 | 0.00 | 0.00 | 1.00 | 1024/1024 |
| sudoku | games | partial | empty=30–50 | 0.10 | 0.00 | 0.82 | 163/169 | 0.04 | 0.00 | 0.93 | 1003/1024 |
| survo | games | exact | board_size=4–5, empty=3–5 | 0.00 | 0.00 | 0.02 | 45/66 | 0.02 | 0.02 | 0.97 | 1019/1024 |
| tower_of_hanoi | games | partial | num_disks=3–7, num_pegs=3–4 | 0.00 | 0.00 | 1.00 | 265/625 | 0.07 | 0.05 | 0.70 | 847/1024 |
| tsumego | games | partial | board_size=9–13, max_stones=15 | 0.05 | 0.00 | 0.00 | 8/8 | 0.03 | 0.03 | 0.95 | 1019/1024 |
| advanced_geometry | geometry | exact+ε | min_coord=-10, max_coord=10 | 0.02 | 0.00 | 0.00 | 20/20 | 0.15 | 0.13 | 0.18 | 401/1024 |
| simple_geometry | geometry | partial* | sides=3–6 | 0.02 | 0.00 | 0.00 | 12/12 | 0.64 | 0.27 | 0.05 | 215/378 |
| course_schedule | graphs | partial* | num_courses=5–10, num_prerequisites=1–2, cycle_length=3–5 | 0.53 | 0.53 | 0.00 | 8/8 | 0.17 | 0.17 | 0.80 | 975/1024 |
| family_relationships | graphs | exact | family_size=4–8 | 0.25 | 0.25 | 0.00 | 8/10 | 0.40 | 0.40 | 0.30 | 100/246 |
| largest_island | graphs | partial* | rows=5–10, cols=5–10, num_islands=0–5, island_size=0–10 | 0.30 | 0.28 | 0.00 | 9/9 | 0.47 | 0.47 | 0.48 | 699/1024 |
| path_star | graphs | exact | degree=3, node_range=100000, path_length=3–5 | 0.00 | 0.00 | 0.00 | 54/127 | 0.55 | 0.55 | 0.45 | 900/1024 |
| quantum_lock | graphs | exact | difficulty=10 | 0.00 | 0.00 | 0.02 | 29/17 | 0.03 | 0.03 | 0.97 | 1014/1024 |
| shortest_path | graphs | exact | rows=5–8, cols=5–8 | 0.00 | 0.00 | 0.00 | 14/17 | 0.30 | 0.30 | 0.70 | 923/1024 |
| acre | induction | partial* | (no curriculum) train=1 | 0.42 | 0.42 | 0.07 | 8/8 | 0.27 | 0.27 | 0.63 | 809/1024 |
| list_functions | induction | partial* | (no curriculum)  | 0.30 | 0.30 | 0.03 | 17/37 | 0.70 | 0.70 | 0.28 | 621/1024 |
| aiw | logic | partial* | task_type_weights=0.3333333333333333/0.3333333333333333/0.3333333333333333, num_entities=6 | 0.02 | 0.02 | 0.02 | 106/250 | 0.68 | 0.68 | 0.25 | 462/1024 |
| circuit_logic | logic | exact | terms=3–5, inputs=2–4 | 0.47 | 0.47 | 0.00 | 8/8 | 0.00 | 0.00 | 1.00 | 1024/1024 |
| knights_knaves | logic | partial | n_people=2, depth_constraint=2, width_constraint=2 | 0.67 | 0.35 | 0.00 | 60/468 | 0.90 | 0.90 | 0.10 | 616/1024 |
| propositional_logic | logic | partial | vars=2–4, statements=2–4, complexity=1–3 | 0.49 | 0.45 | 0.02 | 38/237 | 0.54 | 0.47 | 0.23 | 537/1024 |
| self_reference | logic | exact | difficulty=5 | 0.02 | 0.02 | 0.58 | 926/1024 | 0.00 | 0.00 | 1.00 | 1024/1024 |
| syllogism | logic | partial* | allow_all=True, allow_no=True, allow_some=True, allow_some_not=True | 0.23 | 0.23 | 0.00 | 18/17 | 0.75 | 0.75 | 0.00 | 207/310 |
| zebra_puzzles | logic | exact | num_people=4, num_characteristics=4 | 0.32 | 0.32 | 0.00 | 9/9 | 0.00 | 0.00 | 1.00 | 1024/1024 |
| coin_flip | probability | partial | num_trials=3–15 | 0.34 | 0.27 | 0.00 | 60/411 | 0.89 | 0.85 | 0.00 | 347/587 |

## Example questions

| dataset | example question |
|---|---|
| complex_arithmetic | Divide the complex numbers: (-96.0 + 38.0i) ÷ (-10.0 - 8.0i) |
| intermediate_integration | Evaluate the indefinite integral: ∫ x**3*sin(x) dx When performing calculations, please follow these guidelines: Use same variable symbols as given in |
| polynomial_equations | Determine the real value(s) of q that satisfies: -75*q - 89 = 0 In solving equations, please follow these instructions: 1. Provide all answers as comm |
| polynomial_multiplication | Calculate the following: 23*y*(3*y**2 + 3*y) When performing calculations, please follow these guidelines: 1. Use ** instead of ^ to represent exponen |
| simple_equations | Solve for y: 5*y + 14 = 509 |
| simple_integration | Find the indefinite integral: ∫ 20*x**9 - 9*x**8 + 12*x**5 + 2*x/3 - 1 dx When performing calculations, please follow these guidelines: 1. Use ** inst |
| ab | A::B is a system with 4 tokens: `A#`, `#A`, `B#` and `#B`. An A::B program is a sequence of tokens. Example: B# A# #B #A B# To *compute* a program, we |
| base_conversion | Your task is to convert a number between two different bases. If the target base is > 10, use lowercase letters a-z for digits above 9. Now, convert t |
| binary_alternation | Given a binary string, return the minimum number of character swaps to make it alternating, or -1 if it is impossible. The string is called alternatin |
| binary_matrix | Given a square matrix, your job is to find the taxicab (Manhattan) distance of the nearest 0 for each cell. The output should be a matrix of the same  |
| caesar_cipher | Decrypt this Caesar cipher text: FC GIRXVSTSPMW CSY. Provide only the decrypted text as your final answer. |
| count_primes | Count how many prime numbers there are between 7221 and 7699 (inclusive) ? |
| cryptarithm | Solve this cryptarithm: RSV VRV + GQJ ------ GMJM Each letter stands for a unique digit (0-9). No leading letter can be zero. Provide a comma separate |
| game_of_life | What will this Game of Life board look like after 1 steps of simulation? Assume a Moore neighborhood and wrapping topology. Reply as array of arrays r |
| game_of_life_halting | This is a 'Game of Life' grid. We consider a game halted if there are no cells alive. Will this game halt at or before 20 steps? Assume a Moore neighb |
| graph_color | Please provide a coloring for this graph such that every vertex is not connected to a vertex of the same color. The graph has these properties: Vertic |
| group_anagrams | An anagram is a word formed by rearranging the letters of a different word, using all the original letters exactly once. Your job is to group the anag |
| isomorphic_strings | Two strings are isomorphic if the characters in one string can be replaced to get the second string. All occurrences of a character must be replaced w |
| jugs | You are a police officer. A maniac has planted a bomb next to a public fountain. To defuse the bomb, you must solve a puzzle. The puzzle is solved whe |
| letter_counting | How many times does the letter "a" appear in the text: "story The scene where the village maid discusses interesting philosophical problems with"? |
| letter_jumble | Your task is to unsramble words in a sentence. For each word in a sentence, the letter may have been randomly shuffled. Your task is to unscramble the |
| manipulate_matrix | For the following matrix: 0 1 9 0 1 1 5 3 0 0 0 5 9 7 9 7 2 1 2 1 0 8 7 3 1 8 7 Perform the following series of operations in order: - Identity transf |
| number_filtering | Remove all numbers larger than 2.6 in this list: ['-77', '82.1952', '-93', '-80.32', '-53', '97', '-30.7822', '-3.151', '-70.3', '43'] Return the new  |
| number_sorting | Sort these numbers in descending order: -77.0, 82.2, -93.01, -83.2, -53.0, 97.0, -30.78, -3.2, -70.0, 43.0 Please follow the instruction below: ## 1.  |
| palindrome_generation | Your task is, given a list of letters, to form a valid palindrome. A palindrome is a phrase that reads the same forwards and backwards. If there are m |
| palindrome_partitioning | Given a string, partition it such that every substring is a palindrome. A palindrome is a word that reads the same backward as forward. You may return |
| pool_matrix | Your job is to perform max/average pooling on the given matrix. The stride is equal to the kernel size, meaning there is no overlap between the poolin |
| ransom_note | Given two strings representing a ransom note and a magazine, return True if you can construct the ransom note using the letters in the magazine, and F |
| rotate_matrix | Given a square matrix, your job is to rotate it clockwise. Your output should be a matrix in the same format as the input. Rotate the matrix below by  |
| rotten_oranges | You are given an n x n grid where each cell can have one of three values: - 0 representing an empty cell - 1 representing a fresh orange - 2 represent |
| sentence_reordering | Restore the correct order of words in the following sentence: money enough is to issue. want my research to a All I carry successful |
| spell_backward | Spell this word backward (example: sun -> nus): teemingly |
| spiral_matrix | Given a matrix, your job is to generate a list of elements in spiral order, starting from the top-left element. The spiral order is clockwise, startin |
| string_insertion | Given a string consisting of characters A, B, C, D, and E, your job is to insert a character according to the following pattern: 1. If there is a subs |
| string_manipulation | Your job is to repeatedly transform a string according to a set of rules until no further transformations can be performed, or a state is repeated. Ev |
| string_splitting | There is a dismantling engineer who has old machines A, B, and C. He discovered that he can obtain a batch of new parts X, Y, Z through the following  |
| string_synthesis | There are nine different blocks [A] [B] [C] {A} {B} {C} (A) (B) (C) 1. One [A], one [B], and one [C] can be combined to form one {A}. 2. One [A] and o |
| word_ladder | Transform the word ladder 'CUTS' to 'AHEM' by changing one letter at a time. Provide your answer as a comma-separated sequence of uppercase letters wi |
| word_sequence_reversal | Solve the following problem. Provide you answer as a comma-separated list of words with a space after the comma. Reverse this list of words: story, No |
| word_sorting | Your task is to sort words in ascending or descending order using ASCII/Unicode ordering. Your output should be a comma-separated list of words, e.g.  |
| arc_1d | Find the common rule that maps an input grid to an output grid, given the examples below. Example 1: Input: 0 0 0 0 0 0 0 0 0 0 2 8 0 Output: 0 0 0 0  |
| arc_agi | Find the common rule that maps an input grid to an output grid, given the examples below. Example 1: Input: 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4  |
| rearc | Find the common rule that maps an input grid to an output grid, given the examples below. Example 1: Input: 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2  |
| basic_arithmetic | Calculate -7 + -6 * -9 + 2 / 2. |
| bitwise_arithmetic | Please solve this problem. Assume there is arbitrary bit depth and that there are signed integers. If the answer is negative, reply as a negative valu |
| calendar_arithmetic | Determine if the year 2034 is a leap year. Answer with Yes or No. |
| chain_sum | State the final answer to the following arithmetic problem: 0 + 1 - 9 + 0 + 1 = |
| count_bits | How many 1 bits are there in the binary representation of the number 2075443165? |
| decimal_arithmetic | Please solve this problem to a maximum of 12 significant digits, rounding up from the half. Only reply with the final value. ((0.123+1.376)+(3.880+0.2 |
| decimal_chain_sum | State the final answer to the following arithmetic problem: 0.1 + 1.454 + 9.92 + 0.8 + 1.8 = |
| dice | I have these dice: 1d20, 1d16, 1d5, 1d2. What are the odds of rolling 18 or higher? (Assume that all dice are rolled at once, and that '1d6' represent |
| fraction_simplification | Simplify the fraction 45429/56430 to its lowest terms. Give only the simplified fraction as your final answer. |
| gcd | Find the Greatest Common Divisor (GCD) of these numbers: 120, 8. Give only the GCD as your final answer. |
| gsm_symbolic | A grandfather eats 2 pizzas per day, his wife eats 3 pizzas per day, and their daughter eats 3 pizzas per day. How many pizzas does this family eat in |
| lcm | Find the Least Common Multiple (LCM) of these numbers: 75, 5 |
| leg_counting | Your task is to count how many legs there are in total when given a list of animals. Now, how many legs are there in total if you have 1 human, 13 sna |
| number_format | Your task is to pick the largest/smallest number out of several options. Your output should be only the number of interest. Now, pick the largest numb |
| power_function | Your task is to compute an exponentiation of a number. Compute 932.9071^-7. Return your final answer correct to 3 significant figures. Provide your an |
| prime_factorization | Find the prime factorization of 991. Write the factors separated by × (Example: for 12 the answer would be: 2 × 2 × 3) |
| products | Solve the following multiplication: 0 * 1. Give only the result as your final answer. |
| time_intervals | A conference call began at 1910-10-03 and ended at 1910-10-14. How long was the conference? Answer in D days. |
| bf | This is a BF (Brainf*ck) computer program. What is the output? >[-]>[-]<>++++++++[<++++++++++++>-]<+.+++++++++++.+++..---------.< Respond only with th |
| codeio | You are given a question that requires some input and output variables as follows: Given a system that requires dividing an unsigned integer by a cons |
| color_cube_rotation | A cube has: - a yellow top side - a blue right side - a purple front side - a white left side - a pink back side - a cyan bottom side The cube is rota |
| figlet_font | What word does this say? _ _ __ _ _ /' `\ ' ) ) ( ~-_ /' ' ) ) /' `\ /' ) /' /' /`-__) /' /' /' ._) /' (___,/' /' /' /' ,/' /' (___ /' ; /' /' /' /`-- |
| modulo_grid | Identify the mathematical pattern which defines this grid, then use that pattern to fill in the question marks. Return the entire completed grid as yo |
| needle_haystack | Celik fancies limousines. Abel is committed to graffiti art. Brendan exults in cooking experiments. Tommi-Lee prizes the color fuchsia. Amolpreet ador |
| number_sequence | 5, 5, 10, 15, 25, 40, 65, 105, ? |
| rectangle_count | Your task is to count how many rectangles are present in an ASCII grid. Single rectangles are outlined with a '#', overlapping rectangles (max 2) are  |
| rubiks_cube | You are given a 3x3x3 Rubik's cube. It looks like this: G R B W Y W R R W O R Y B B R B B Y O G W Y R W G G B Y O O W B R Y G W O O G W Y Y O Y R B B  |
| boxnet | You are a central planner tasked with directing agents in a grid-like field to move colored boxes to their corresponding color-coded targets. Each age |
| countdown | Calculate 126 using all of these numbers: 3, 3, 3, 45, 83. Each number may be used at most once. Final answer format instructions: 1. Provide your sol |
| emoji_mystery | The following emoji is encoded with a sentence. Decode the following sentence from the emoji: 💔󠄹󠅤󠄐󠅙󠅣󠄐󠅔󠅕󠅜󠅙󠅗󠅘󠅤󠅖󠅥󠅜󠄐󠅤󠅘󠅥󠅣󠄐󠅤󠅟󠄐󠅤󠅑󠅛󠅕󠄐󠅒󠅢󠅕󠅑󠅛󠅖󠅑󠅣󠅤󠄐󠅏󠅤󠆳󠆚󠅤󠅕󠄝󠆳󠆐󠄝󠅤󠆳󠆚󠅤󠅕 |
| futoshiki | Solve the following 7x7 Futoshiki puzzle: _ 4 _ _ 5 7 _ _ _ 4 _ 2 _ 3 ∨ _ _ _ _ _ 3 _ 3 5 _ _ _ _ _ _ _ 5 2 1 _ 4 6 _ 1 _ _ _ 2 _ _ 2 _ _ _ 6 Ensure y |
| kakurasu | You have a 5 x 4 Kukurasu puzzle grid. Your goal is to place 1s in the grid so that the weighted sums match the given constraints: row sums [2, 4, 3,  |
| knight_swap | Knight Swap Challenge: ``` A B C D ---------------- 3 / / B / . / B / ---------------- 2 / . / / . / . / ---------------- 1 / w / / w / . / ---------- |
| mahjong_puzzle | There are several letter cards, and the game rules are as follows: 1. Initially, there are 13 cards. 2. Each time, a new card is added, and a result i |
| maze | Navigate from '0' (start) to 'b' (goal): ``` UUUUUUUUUU UeeeeeeeUU UeeUeee0eU UeUUeUeUeU UUeeeUeeUU UeeebUUeeU UeeeUeeeUU UeUeUUeUeU UeUUeeeUeU UUUUUU |
| mini_sudoku | In 4x4 Mini Sudoku: - Each row must contain each number from 1-4 exactly once - Each column must contain each number 1-4 exactly once - Each 2x2 subgr |
| n_queens | Your job is to complete an n x n chess board with n Queens in total, such that no two attack each other. No two queens attack each other if they are n |
| puzzle24 | Make 24 using 7, 6, 4, 7. You can only use each number once. You can use the operators +, -, *, /. Final answer format instructions: 1. Provide your f |
| rush_hour | Move the red car (AA) to the exit on the right. Specify moves in the format: 'F+1 K+1 M-1 C+3 H+2 ...' where the letter is the vehicle and +/- number  |
| sokoban | You are going to solve a 'sokoban' puzzle. * - The player % - The player on a goal @ - A box X - A goal $ - A box on a goal + - A wall - - An empty po |
| sudoku | Solve this Sudoku puzzle: 9 4 _ 1 2 8 _ 5 7 6 5 7 _ 9 _ _ _ _ _ _ _ 6 7 _ 9 4 _ _ _ _ 4 3 7 2 1 5 _ 3 4 _ _ _ 7 _ _ 5 _ 1 _ 6 _ _ _ _ 7 _ 5 2 1 3 4 6  |
| survo | Solve this 5*5 matrix puzzle: 0 2 1 0 6 2 6 4 1 13 1 1 6 8 16 8 0 2 3 16 12 12 13 14 51 where 0 represents empty cells that need to be filled. The las |
| tower_of_hanoi | Solve the Tower of Hanoi problem with 6 disks and 3 pegs. Move all disks from Peg 1 to Peg 3 following the rules: - Only one disk can be moved at a ti |
| tsumego | I have a Go problem for you. Black moves next - can you capture some of the white stones? ``` A B C D E F G H I J K L 12 . . . . . O . . X . . . 11 X  |
| advanced_geometry | Consider triangle ABC with coordinates A=(-7, -10), B=(-8, 8), and C=(-9, -8). Compute the radius of its incircle. For all geometry problems: 1. Give  |
| simple_geometry | Consider a convex 6-gon whose first 5 interior angles are: 39.0°, 11.0°, 33.0°, 159.0°, 18.0°. Determine the measure of the remaining angle.Return onl |
| course_schedule | There are a total of 8 courses you have to take, labeled from 0 to 7. You are given the following list of prerequisites, where prerequisites[i] = (a_i |
| family_relationships | Joseph is married to Mary. They have a child called Robert. Robert is married to Susan. They have a child called Noah. David is married to Sage. They  |
| largest_island | You are given the following 8 x 5 binary matrix grid: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 An island is a g |
| path_star | Find a path from the start node to the goal node in the following path-star graph. Respond with only the sequence of node labels in the path, includin |
| quantum_lock | In front of you are some buttons, a light, and a number. The light will toggle between red and green whenever you press a button. Each button performs |
| shortest_path | Your task is to find the shortest path from the start to the destination point in a grid. The grid is represented as a matrix with the following types |
| acre | You are a researcher studying causal relationships using Blicket experiments. In these experiments, certain objects (called 'blickets') have the hidde |
| list_functions | You are an expert at inductive reasoning. Generate an output corresponding to the given input. The output is generated by applying the same rule that  |
| aiw | Jessica has 1 male colleagues and she also has 1 female colleagues. These are all colleagues that Jessica has. All these mentioned persons around Jess |
| circuit_logic | Below is a randomly generated logic circuit. A: ─────────────┐ B: ───────────┐ │ C: ─────────┐ │ │ D: ───────┐ │ │ │ E: ─────┐ │ │ │ │ F: ───┐ │ │ │ │ |
| knights_knaves | A very special island is inhabited only by knights and knaves. Knights always tell the truth, and knaves always lie. You meet 2 inhabitants: Victoria, |
| propositional_logic | The following question is a propositional logic reasoning question. In the question we provide a list of premises. The task is to infer a correct conc |
| self_reference | Given the truthfulness of these statements, please tell me the number of possible solutions: - Statement 1: 'At least 1 of these 7 statements are true |
| syllogism | Consider these statements: 1. All ants are students 2. All students are mortals Does it logically follow that: Some students are ants? (Answer Yes or  |
| zebra_puzzles | This is a logic puzzle. There are 4 houses (numbered 1 on the left, 4 on the right), from the perspective of someone standing across the street from t |
| coin_flip | What is the probability of getting at least 3 heads in 15 fair coin flips? |

## Notes

- Fail to generate at the default config: `composite` (AssertionError: Must specify at least one dataset).
- Slow to generate (more than 10 s for 60 instances, measured with 24 datasets generating in parallel): `futoshiki` (28s), `knight_swap` (26s).
- Prompts longer than 1500 tokens (max prompt tokens): `arc_agi` (12778), `codeio` (3322), `rearc` (3358), `rotten_oranges` (2018).
- Junk answer `zzz` scores at least 0.5 (scorer unusable as a reward without a fix): `game_of_life_halting` (1.00).
- A missing answer still scores above 0 (score): `countdown` (0.01), `fraction_simplification` (0.01), `puzzle24` (0.01).
- The stored gold answer scores below 1 (mean; the scorer checks validity and `answer` is not a valid solution string): `boxnet` (0.00), `graph_color` (0.00), `propositional_logic` (0.00), `rubiks_cube` (0.01), `rush_hour` (0.01).
- `score_answer` raised an exception on a perturbed gold answer (guard reward calls): `prime_factorization`.
- Request or scoring errors during calibration (count; see the jsonl `error` field): `arc_agi` (8), `game_of_life` (1).
