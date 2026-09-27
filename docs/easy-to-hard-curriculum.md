# Easy-to-hard environments

New uploads use four explicit task contracts: foundation, guided practice,
application and composition. Each contract specifies the number of concepts,
scaffolding and edge cases appropriate to its level. The generator receives the
contract; a separate blind reviewer must approve both the answer and stage fit.
These are designed difficulty levels, not claims about measured learner ability.

A 100-problem environment contains 20 training and five held-out problems at each
level. A 20-problem environment contains four training and one held-out problem
per level. Exported task records include level, prerequisites and requirements.
The portable runtime retains private answers and separate train/heldout access.
Previously frozen uploads retain their original split and can still resume.

The replacement coding dataset progresses through single-expression bug repair,
one bounded DFS on small trees, independent IDA* returning path/cost, and fully
instrumented IDA* with cycles and edge cases. All 3,000 private/development examples
are checked by two reference implementations; planted foundation bugs and constant
wrong programs must fail. Development and private inputs are disjoint, as are
training and held-out inputs. The holdout measures new input sets and variants
within practiced skill families, not unseen algorithms.

The new experiment starts with original Qwen2.5-7B, a fresh rank-8 LoRA adapter,
and a frozen 20-task baseline. Twenty-four planned updates allocate six updates
per level in order. Each update compares two three-attempt execution-feedback
trajectories. Reward remains half private test accuracy plus half private field
accuracy. Held-out results do not control the schedule, rewards or checkpoint
selection. The final completed adapter is evaluated with the same heldout and
budgets, reporting both first-attempt and repaired success by level. A separately
recorded two-hour compute cap reserves forty minutes for final evaluation.

New interactive LEGO datasets increase the missing-piece count from one to four;
the final level also requires correcting a misplaced piece. Frozen older datasets
and published scores are unchanged.
