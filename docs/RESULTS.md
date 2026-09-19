# Results, failures and interpretation limits

These are internal controlled experiments, not an independently reproduced benchmark or peer-reviewed result. Percentages below require the complete correct target **and EOT**, unless stated otherwise. The three seeds vary lesson banks/order from one formed model; they are not independent formation replicas. Small differences below the involved seed spread are not ranked.

## 1. Acquisition and preservation-assisted consolidation

| Seed | After acquisition: main | Continue fitting: main | Preserve while fitting: main | Preserve: held wording | Preserve: old examples |
|---|---:|---:|---:|---:|---:|
| 79701 | 256/256 | 242/256 | 256/256 | 320/320 | 48/48 |
| 79702 | 254/256 | 236/256 | 256/256 | 320/320 | 48/48 |
| 79703 | 256/256 | 234/256 | 256/256 | 320/320 | 48/48 |

The preservation condition exceeded continued fitting by 7.29 percentage points on average, above their 3.125-point range. Its 0.26-point advantage over stopping after acquisition was below the 0.78-point range. **Consolidation is not shown to beat stopping.** Both further-fitting arms met their original declared gates.

These phases revisit the same reading skill. They are not a second independent new skill or a long-lifetime durability test. All acquisition endpoints scored 320/320 on held wording. Missing-owner handling remained weak: preservation endpoints correctly returned UNKNOWN on 24/80, 7/80 and 17/80; they returned a known positive value on 59/80, 72/80 and 63/80. Two-record pair success was 0/64, 1/64 and 0/64. Successful reading did not establish record-membership or multi-owner binding.

The distributed checkpoint is the previously frozen demonstration default from seed 79701's preservation condition, **not** the subsequent best checking model.

## 2. Counterfactual single-record checking: development only

Three arms continue from matched acquisition parents. POS practices positive reading. EASY practices relation absence. CHECK holds a record fixed and changes whether the queried owner matches it. EASY/CHECK match target exposure and losses. A pair passes only when both present and absent queries pass, preventing an always-copy or always-UNKNOWN strategy from succeeding.

| Seed | POS pairs | EASY pairs | CHECK pairs | CHECK minimum primary cell | CHECK unseen-wording pairs | Minimum wording cell |
|---|---:|---:|---:|---:|---:|---:|
| 79801 | 0/256 | 0/256 | 247/256 (96.48%) | 90.625% | 111/256 (43.36%) | 0% |
| 79802 | 0/256 | 0/256 | 246/256 (96.09%) | 90.625% | 88/256 (34.38%) | 0% |
| 79803 | 0/256 | 0/256 | 253/256 (98.83%) | 96.875% | 117/256 (45.70%) | 0% |

Mean CHECK primary = 97.14%; mean unfamiliar wording = 41.15%. The primary contrast exceeds the largest involved 2.73-point seed range. **The original joint gate failed in all three CHECK seeds because of wording transfer. Confirmation remains unscored.** This is strong development evidence for a bounded teaching effect, not a confirmed general skill.

All nine endpoints scored 100% on separately audited subsets in their own native training format. CHECK ended at 48/48 old replay examples in each seed; it retained 48, 48 and 47 previously correct examples and recovered one in the third seed. FineWeb NLL was 3.364130/3.364120/3.364530, Cosmopedia 2.270322/2.270858/2.270532, against reference 3.360978/2.266411.

**Important counterevidence:** CHECK lost 145/256, 168/256 and 138/256 formerly correct positive answers on unfamiliar wording. Good replay and prose scores did not protect an omitted behavior surface. Pooled preservation claims must include this churn.

Two-record paired accuracy stayed 0% in every CHECK seed. In the code-only subset all three models returned the first record's value on 128/128 queries. Mention-only pairs, where an owner had a rank but no code, reached 6.25%, 7.8125% and 0%.

All nine endpoints reconstructed in a fresh process with exact protected parameters/topology. Fixed-endpoint optimizer time totaled 20.86 minutes on RTX 3070 Ti; this excludes evaluation, loading, hashing and interrupted attempts. Peak allocated CUDA memory was about 2.08 GiB, not total process or reserved VRAM. This is not a speed comparison with another architecture.

## 3. A proposed repair was rejected

A read-only construction/placement audit used 72 fresh identifiers and 2,880 queries across parent/CHECK states and three worlds.

| No-modifier construction | 79801 | 79802 | 79803 |
|---|---:|---:|---:|
| “What code belongs to OWNER?” | 100% | 100% | 100% |
| “Retrieve the code for OWNER.” | 25% | 0% | 4.17% |

Moving modifiers did not consistently rescue the weak construction. All four declared restoration contrasts were smaller than their involved seed spread. **The placement repair candidate was not supported.** Syntax, vocabulary, token count and position were not uniquely disentangled; this is not proof of a localized syntax mechanism. No production prompt rewrite follows.

## 4. Evidence files and exclusions

`evidence/owner_check_results.json` retains all development arms' summaries and item-level churn aggregates. `evidence/query_construction_results.json` contains the read-only factorial. `evidence/generation_examples.json` retains deterministic example selections, including failures, with exact tokens. These are exported historical evidence, not fresh test data.

The small release does not contain all full run checkpoints, full generated-query logs, raw corpus windows or the from-scratch formation runner. Consequently it supports inference, local-update engineering and result inspection, **not one-command independent reproduction of every historical table**. The omissions are intentional and documented, not replaced by illustrative data.

## 5. What remains unestablished

Dependable open-ended speaking; construction-invariant checking; reliable multi-record ownership; generalized parametric membership; durable accumulation of distinct skills; corrections in this released body; automatically complete preservation contracts; growth superiority; or an advantage over parameter/compute-matched conventional models.
