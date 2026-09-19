# Architecture and mechanism standing

## Implemented computation

The released checkpoint has 151,905,024 parameters, 16 decoder blocks, width 768, 12 attention heads, context limit 1,024 tokens, 50,257 usable GPT-2 token IDs, and 50,304 padded embedding rows. Input embeddings and output projection are tied.

Each block applies layer-normalized causal rotary attention, followed by a residual computational graph. Four vertex modules each use a bias-free width→736→width GELU transform. Directed connections run from lower-index to higher-index vertices. With coordinates p, a connection coefficient is `exp(-||p_i-p_j||²/2)`, multiplied by 0.1 in the context sum. Coordinates are explicit geometry metadata, not trainable semantic embeddings in this checkpoint.

For a declared face, each participating module output is RMS-normalized, passed through tanh, and multiplied elementwise with the others. A separate GELU up/down module transforms this product. Two triangular modules use rank 32; one tetrahedral module uses rank 64. Vertex and face outputs are summed into the residual stream.

The face product introduces an explicit higher-order interaction. It does not prove that geometric coordinates represent semantic dimensions. The two scalar reference bounds stored with the geometry are metadata, not neural extremum units.

## Adaptation and preservation

The recent language experiments update blocks 12–15: 28,317,696 trainable scalars. Earlier blocks, embeddings, output head, final normalization and topology remain exact. Frozen parameters do **not** imply unchanged output behavior; preservation requires functional evaluation.

The historical preservation objective combines new-task CE, old-example CE and forward KL from a declared frozen reference on a declared input distribution. Both the reference **and the input distribution** matter: generic prose KL cannot be assumed to protect old factual queries absent from its inputs.

## Experimental structural APIs

`Model.birth` supports simplex subdivision, dimensional lifting and paired additions. New residual modules have zero output projection, preserving the initial function; subsequent training may change it. `Model.intervention` temporarily disables vertices/faces or changes coordinates and restores the original structure on exit. `project` and `round_parameters` coarsen selected parameter values; neither guarantees preservation or reduces FP32 storage by itself.

These APIs are available for ablation research. Tiny-model tests cover zero-output insertion and lesion restoration. They are **not an autonomous growth controller**, and language-scale benefits of growth, rounding or structure retirement are unestablished. The distributed model contains no added modules.

## Scope of comparisons

This is the geometric residual decoder, not the earlier two-stream prototype explored elsewhere in the research history. No result from that different model is used as a guarantee here.

There is no matched conventional-model control demonstrating better accuracy, training efficiency, inference speed or lifelong retention. Adding products and modular transforms may add execution overhead. Published throughput comparisons require matched hardware, batch/context lengths, precision and training histories; none is claimed in this release.
