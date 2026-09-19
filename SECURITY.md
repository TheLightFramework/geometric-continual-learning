# Security and operational limitations

Use a maintained Python/PyTorch environment. Load only trusted checkpoints, even with tensor-only deserialization. JSON manifests and hashes detect accidental changes but are not signatures. This release is not hardened against hostile files, concurrent filesystem mutation or malicious evaluators.

The transaction store assumes a trusted local process and single writer. It cannot certify semantic correctness or protect omitted behavior. It provides version selection and integrity checks, not a security boundary or production authorization service.

Do not expose the model as an unrestricted public service or let its output authorize external actions. No authentication, rate limiting, content filtering or privacy guarantee is provided. Report non-sensitive bugs through repository issues after publication; arrange a private channel with the maintainer before sharing sensitive details. No monitored private security address is asserted here.
