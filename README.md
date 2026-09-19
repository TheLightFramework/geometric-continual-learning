# Geometric Continual Learning — Research Prototype

A 151M-parameter geometric residual language model, controlled local-update experiments, and an auditable **acquisition → consolidation → evaluation → accept/reject → rollback** workflow.

**Version 0.1.0 · MIT project code · Experimental research release · Not a general assistant**

This release makes a partial research result usable and inspectable. It does **not** claim to solve lifelong learning, reliable factual memory, autonomous obligation discovery, or architectural superiority over conventional Transformers.

## What is supported?

- A formed language model can acquire restricted reading skills through updates to its last four blocks, while protected parameters remain exact.
- Preservation-assisted further fitting retained the tested old reading examples better than unregularized continued fitting. It did **not** establish a meaningful advantage over simply stopping after acquisition.
- A subsequent development study improved single-record owner checking to **97.14% paired exact accuracy**, but unfamiliar question constructions reached only **41.15%**, with a **0% minimum cell**. Its joint gate failed; confirmation was not run.
- Candidate checkpoints can be evaluated without replacing the reference model. Hash-bound review and rollback are implemented and fault-tested.

The answers in these reading experiments are present in the prompt. **Reading supplied records is not the same as remembering facts with the records absent.** See [Results](docs/RESULTS.md), which includes failures, seed ranges and item-level losses.

## Start here

Requires Python 3.11+ and PyTorch. The tested environment is recorded in `evidence/RELEASE_VALIDATION.json`; dependency ranges are not a claim that every combination has been tested. Install an appropriate PyTorch build for your device using its [official instructions](https://pytorch.org/get-started/locally/), then:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install .
python -m unittest discover -s tests -v
```

Tests use tiny random models and do not download weights or datasets.

### Inspect and generate

Obtain the **separate model asset** supplied with this release. It contains `weights.pt` and `MANIFEST.json`; keep them together in `weights/reader-consolidated-79701/`. The source repository deliberately excludes large checkpoints. See [Artifacts](docs/ARTIFACTS.md).

```sh
python -m gcl inspect --weights-dir weights/reader-consolidated-79701
python -m gcl generate --weights-dir weights/reader-consolidated-79701 --device cpu --prompt "Question: Notes: Mira has code 42. What code belongs to Mira? Give only the code. Answer:"
```

`--device cuda` is available. Loading starts on CPU, checks file/tensor/topology hashes, then transfers to the requested device. GPT-2 tokenization is bundled for offline use. Generation is greedy, reports termination, and refuses oversized contexts rather than silently truncating them. It is a completion interface, not a chat template.

### Run a complete local-update demonstration

```sh
python -m gcl.experiment --weights-dir weights/reader-consolidated-79701 --lesson examples/lesson.json --out runs/demo-001 --device cuda
```

The output directory must be new. The command seals the lesson/contract before fitting, runs two acquisition and two consolidation updates on a clone, evaluates held authored reading examples and replay examples, checks the frozen partition, saves/reloads the candidate, and writes a **review-only** decision. No automatic promotion, downloads or cloud jobs occur. A failed decision is an expected, useful outcome—not a reason to lower its contract.

This tiny example is an **engineering demonstration**, not reproduction of the historical study, not a generic-language retention test, and not evidence of new factual memory. Its reference is the loaded parent and its KL inputs are the two synthetic strings in the lesson. See [Workflow](docs/WORKFLOW.md) for the distinction and for explicit application/rollback.

## Architecture and unresolved questions

The decoder uses rotary attention, four residual modules per block, geometry-weighted directed connections, explicit triangular/tetrahedral product interactions, and a tied output head. Geometry and growth operations are implemented; growth did not produce the positive language results reported here. [Architecture](docs/ARCHITECTURE.md) separates implemented mechanisms from validated benefits.

Priorities for independent research: question-construction transfer, multi-record binding, broader retention coverage, sequential distinct-skill acquisition, and matched conventional architecture controls. [Roadmap](docs/ROADMAP.md).

## Publication and contribution scope

- [Model card](docs/MODEL_CARD.md)
- [Reproducibility and evidence boundaries](docs/REPRODUCIBILITY.md)
- [Data provenance and third-party terms](docs/DATA_AND_LICENSES.md)
- [Contribution guide](CONTRIBUTING.md) and [security limits](SECURITY.md)
- [Citation](CITATION.cff)

Led by Jean Charbonneau using AI-assisted implementation and multiple independent model critiques. These critiques informed experiments; they are **not independent human peer review**. No institutional affiliation, external endorsement or benchmark leadership is claimed.
