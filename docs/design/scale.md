# A larger base model

Everything up to 27 November used Qwen3-1.7B. Its results came with one caveat that no further pilot at that size can remove: a model that small may know only one way of solving a problem, and then the only thing a lineage can make its own is manner. This stage repeats the decisive questions on a much larger model of the same family, Qwen3.8-27B, with intermediate sizes used where the 27B is too slow.

## What the small model said

- A lineage's own direction away from the base model is real, inherited and easy to strengthen, and it consists of manner and length. Wherever two members solved the same problem they solved it the same way.
- Distance from the base model is a cost: across members and across three interventions, moving further meant transferring less.
- The one rule that helped pulled a member's correct answers back toward the base model.

The thesis needs the opposite of the first point. Bias amplification can only produce a way of reasoning if the base model holds several, so that a lineage has something of substance to settle on.

## Questions, in the order they are asked

1. **Is there anything but manner to amplify?** Take problems both models can solve and compare each model's own correct answers to the same problem. How often do two of them use a different method, as opposed to the same method worded differently? And if the model is shown one of its correct solutions and asked for a different method, how often does it produce one that is also correct? The first measures the variation that training on one's own attempts could select from; the second measures what a seed could unlock. This needs no training.
2. **Does competence still sit where behaviour is typical?** For the base model alone, among several attempts at one problem, are the correct ones the more probable ones under the model itself? At 1.7B that relation is what made moving away costly. It can be read from the model's own attempts before any member is trained, and compared between sizes.
3. **What do members become?** Plain training of a few lineages on the large model's own learnable problems: do they diverge in method and not only in manner, and does a lineage keep its method when cloned?
4. **Does amplifying a lineage's own direction now pay?** The taste-shaped reward in both directions against plain training, from identical weights, as in the 1.7B pilot. The outcome to watch is whether the members pushed along their own direction change method, and whether their accuracy on unseen sources still falls with their distance from the base model.
5. **Taste at the level of method.** If the first question is answered yes, the reward can be shaped by method instead of by wording: a member is paid more for a correct answer that uses the approach it has used most, so that an early accidental preference feeds itself. This is the mechanism the project set out to find, and it is only meaningful on a model that has more than one method.

A clear no to the first question on the 27B would mean the 1.7B result is not about size, and the stage would end there with that conclusion.

## How method is judged

A method is the idea that produces the answer: the technique in a math solution, the algorithm and data structure in a program, the line of argument in a logic problem. Wording, layout, order of exposition, level of detail and whether the answer is checked are not method.

A judge model reads the problem and two correct solutions, without knowing which model wrote them, and names the approach of each before deciding whether they are the same, a variant of one idea, or different. The same judge reads both models' answers. Its verdicts are checked against independent readers on a sample, and for programs against a mechanical comparison of the constructs used.

## Environment

The problems the 1.7B solves sometimes are mostly solved every time by a model sixteen times its size, so the learnable middle has to be found again: a census of the large model over the training sources, weighted toward the problems the small model never solved, defines its own strata. The verifiers, the answer formats and the length rule are unchanged. Selection and blind items stay as they are; no model, at any size, is run on blind items before a harvest is committed.

## Compute

The stage runs within the cap of four GPUs. Which of the model's forms is trained, how sampler and trainer are laid out over cards, and what a training step costs are being measured before the training pilots are sized; the measurements and the resulting plan are recorded in the logbook.
