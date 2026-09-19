# Acquisition, consolidation and verified checkpoint transitions

## The experiment contract

1. Keep a tensor-only parent checkpoint immutable.
2. Reserve the lesson, evaluation inputs, trainable names, losses, budgets and gates before fitting.
3. Evaluate the parent. Fit a disposable clone.
4. Acquisition uses task CE; consolidation adds explicit replay CE and functional KL.
5. Evaluate the candidate on the declared acquisition and retention surfaces. Record errors, not just scores.
6. Verify protected tensors, parent identity, topology and serialization.
7. Bind an evaluation receipt to the exact candidate, parent and contract.
8. Review; reject or explicitly select a new version. Rollback selects a verified ancestor.

Acquisition and consolidation are learning phases, not biological states. The demo's four optimizer steps do not establish long-term memory.

## What the bundled demonstration actually does

`examples/lesson.json` fixes two acquisition and two consolidation steps, AdamW 3e-6, zero weight decay, gradient clipping at 1, blocks 12–15, FP32 and math SDPA. Consolidation loss is task CE + 1×replay CE + 4×forward KL from the **loaded parent**, evaluated on two **authored synthetic strings**. It is not the historical guided reference with corpus replay. There is no generic prose-preservation claim from this demo.

Prompt plus answer/EOT targets are isolated in individual causal rows; prompt tokens carry no answer CE. Acquisition evaluation prompts are disjoint from training prompts. Retention examples are explicitly replayed: their post-fit success is not held-out transfer. Full target strings and termination are scored. Every generated sequence is saved. Loss and pre-clipping gradient norm are logged each step.

The output contains the contract, input lesson, baseline, phase evaluations, tensor-only candidate, trajectory, final evaluation, and transaction decision. Thresholds are sealed before fitting. The default parent stays active even when the candidate is eligible. Use a new output directory for every run; there is no hidden checkpoint selection, resume or early stopping.

## Explicit research-sandbox selection and rollback

Only after inspecting a complete run:

```python
from gcl.transactions import Store, read
store = Store('runs/demo-001/store')
before = store.current()
decision = read('runs/demo-001/DECISION.json')
result = store.review(decision['candidate'], apply=True)
print(result['applied'], result['failed'])  # failed gates still prohibit selection
if result['applied']:
    store.rollback(before)
assert store.current() == before
```

This updates an atomic JSON pointer, never the original weights. The filesystem store records absolute paths to artifacts: keep the run and its references intact. It is not a self-contained movable database. Artifact mutation is detected at access time. A stale writer lock requires inspection; it is not automatically removed. A recorded event before pointer replacement is an intent, not proof of commit. `CURRENT.json` is authoritative.

## What verification cannot do

The evaluator is trusted and the contract is finite. An omitted skill may fail even when every check passes. Checksums do not authenticate a maliciously substituted manifest. This is a local single-writer prototype, not a secure multi-user deployment or crash-durable distributed database. Ordinary atomic pointer replacement does not promise filesystem-level durability after power loss.

To extend the science, reserve fresh development and confirmation data, expose no confirmation queries to losses or recipe selection, report all seeds and item-level churn, and declare the reference **plus query distribution** for each preservation term. Do not automatically widen the update region or lower gates after a failure.
