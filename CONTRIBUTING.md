# Contributing

This is an experimental release, and careful criticism is welcome. Start with the results and limitations rather than assuming a general memory system exists.

For bugs: include the commit/release identifier, Python/PyTorch/device versions, command, manifest hashes and a minimal reproduction without private data. Do not attach credentials or whole checkpoints to issues.

For experiments: state the question, control, trainable scope, data provenance, development/confirmation separation, fixed budget, metrics and stop rule before fitting. Keep old evidence immutable. Report negative results, every seed, template floors and item-level churn. Distinguish reading supplied context from recalling absent information.

Run `python -m unittest discover -s tests -v`. Tests do not need the model artifact. Functional model experiments require separately downloaded weights and more resources. Document changes to numerics; do not silently label a refactor as bitwise equivalent.

Project contributions are intended under MIT. Preserve third-party notices and disclose AI-assisted work accurately. No claim of novelty, external endorsement or independent peer review follows from model-generated critiques.
