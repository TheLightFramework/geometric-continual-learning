# Data, software and license boundaries

## Project-owned material

The project-authored software, documentation, synthetic lesson fixtures and project checkpoint are offered under the repository's MIT license, copyright 2026 Jean Charbonneau. Existing notices are preserved. This is not a blanket relicensing of dependencies, tokenizer material, training corpora or arbitrary model outputs, and is not a legal clearance opinion.

## Training sources, not redistributed here

- **FineWeb**, Hugging Face: the dataset card declares ODC-By 1.0 and discusses Common Crawl terms. [Official card](https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/main/README.md).
- **Cosmopedia**, Hugging Face: the dataset card declares Apache-2.0. [Official card](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia/blob/main/README.md).
- Later guidance/reading exercises were authored synthetic lessons. Public examples in this release are synthetic, not private correspondence.

The raw corpus reservoirs, protected prose documents, public-corpus record IDs and full acquisition pipeline are not distributed. The exact upstream corpus revisions are not reconstructed by this release. Corpus licenses and webpage-level rights should not be conflated. Consult the source cards and applicable terms before reconstructing or redistributing datasets; the checkpoint license does not certify rights in every underlying document.

## Bundled tokenizer material

`gcl/assets/gpt2.json` is an offline export of the GPT-2 encoding used by tiktoken. It is not a new vocabulary trained for this project. Upstream notices are included in `third_party/TIKTOKEN_LICENSE.txt` and `third_party/GPT2_LICENSE.txt`, retrieved from [tiktoken](https://github.com/openai/tiktoken/blob/main/LICENSE) and [GPT-2](https://github.com/openai/gpt-2/blob/master/LICENSE). The GPT-2 notice is titled Modified MIT; do not relabel it as the project's standard MIT text.

PyTorch, NumPy, SciPy and tiktoken are installed as dependencies, not vendored library implementations. Their own licenses apply. The extracted attention and geometry classes originate in this project's historical implementation; their source hashes and extraction scope are recorded in `evidence/SOURCE_LINEAGE.json`.

## Excluded private material

No raw model-council transcripts, personal messages, machine usernames/absolute archive paths, authentication material, cloud credentials or original working directories belong in the publication. The preparation journal and private path mapping remain outside the repository.
