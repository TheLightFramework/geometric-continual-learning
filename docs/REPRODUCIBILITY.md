# Reproducibility scope

## Included and executable

1. The inference/structural model implementation, offline tokenizer, exact tensor/topology checkpoint hashes, and separate tensor-only model asset.
2. A portable acquisition/consolidation example with pre-fit contract, logged losses, generated evaluations, frozen-partition checks, serialization and a review-only decision.
3. CPU tests for masked local updates, cache behavior, context limits, growth initialization, lesion restoration and transaction fault handling.
4. Seed-level historical result exports and selected exact-token failures. Evidence exports are mechanically relabeled; source and output hashes identify the transformation.

## Not included or not established

The release is not the entire laboratory archive. It does not include full from-scratch formation scripts, training corpora, every starting/endpoint checkpoint, optimizer states or every original generated query. It cannot reproduce all historical tables with one command. The distributed demonstration checkpoint is **not** any of the later CHECK endpoints, so its outputs must not be presented as reproducing their 97.14% score.

A clean install supports the public demonstration and engineering tests. Historical experiments remain evidence with explicitly limited public rerun coverage. Wider independent replication is a priority, not claimed as completed.

## Historical counterfactual-checking recipe

Three data/order seeds 79801–79803 used paired acquisition parents. Each world had 32 train and 64 held six-letter owners screened for three leading-space GPT-2 tokens. Values varied across presentations. POS used eight positive queries per optimizer step; EASY and CHECK used four positive and four negative queries. EASY's negative relation was rank instead of code; CHECK changed the queried owner with the code record fixed. They matched supervision amounts but were not token-identical causal manipulations.

All fits used 256 updates, eight task queries per update, query-mean answer/EOT CE, microbatch two, FP32, deterministic algorithms and TF32 disabled. AdamW: LR 3e-5, 16-step linear warmup, betas (0.9,0.95), epsilon 1e-8, zero decay, clip norm 1 after all losses. Only blocks 12–15 changed. Preservation: 4×forward KL from the guided parent on two 128-position TRAIN prose windows per step, plus 1×CE on two historical authored lesson examples. POS had 4,096 task targets; EASY/CHECK had 5,120. Prose KL covered 65,536 positions per fit. No held-owner questions entered optimization.

Success required main present/absent exact ≥90% each, paired exact ≥80%, minimum primary cell ≥70%, held-wording and held-marker paired exact ≥70% each, retention of ≥95% of previously correct old examples, protected-source NLL ≤ guided-reference +0.05, and exact integrity/reconstruction. All three CHECK worlds had to pass before confirmation. They did not; no threshold was lowered.

The interrupted third-world attempt was preserved and rerun under a versioned wall-budget amendment only. Its 256 numerical training steps and saved evaluations reproduced the interrupted continuation control exactly, excluding wall time. Six completed endpoints were reused. This operational recovery did not change the failed scientific gate.

## Fresh experiments

All shipped prompts and historical banks are now public development/regression material. Do not use them as unseen confirmation. Reserve fresh owners, constructions, ordering seeds and exact gates, hash them before fitting, and report every seed with its denominator, template floor and item-level churn. Comparisons smaller than seed spread should not be ranked. Three lesson worlds from one model do not estimate variance across independent formation runs.

## Numerical portability

The model asset uses FP32 tensors. Same-stack hash identity and exact reconstruction are checked. Cross-device/kernel behavior is not promised bit-identical. Declare hardware, PyTorch/CUDA versions, precision and SDPA backend. The example uses math SDPA explicitly and records its environment; inference may use the device's standard PyTorch attention dispatch.

`requirements-tested.txt` records this machine's scientific dependencies, with the CUDA build in the validation receipt. Broad package requirements exist for installability, not universal numerical certification. The generated GitHub Actions configuration has not run on GitHub before the repository is published.
