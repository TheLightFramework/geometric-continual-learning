# Source and model assets

The Git repository contains source, an offline tokenizer, contracts, documentation and selected historical evidence. The model tensor file belongs in a **separate release asset**, not a normal Git commit.

Expected model directory:

```text
reader-consolidated-79701/
    MANIFEST.json
    weights.pt
    LICENSE
    DATA_AND_LICENSES.md
```

The manifest is also included under `artifacts/`. Expected weights:

- Size: 607,708,859 bytes.
- SHA256: `9340642c12d36d67516c6b0b92a1f7a828161013282d5b13d90c56d8347afc2a`.
- Tensor payload fingerprint: `95cb0d93b8fb1121b79b1fb34638bc80179e08eed502380724ac493cd454af64`.
- Topology fingerprint: `4cd800883bd4d966a779653381673d3a2c245ded19fa83fa5df09a3b7bf2ddc1`.

Only obtain files from a release you trust. The loader checks file bytes, reconstructed tensors and topology; it uses `torch.load(weights_only=True)`. The source tree's `RELEASE_MANIFEST.json` can be verified with `python tools/verify_release.py`. Neither local checksum substitutes for a separately trusted publisher identity.

No download URL is invented before publication. The maintainer should attach the prepared model archive to the same GitHub Release as the source tag. Users can obtain the source without downloading the model. Weights and optimizer outputs are ignored by Git.
