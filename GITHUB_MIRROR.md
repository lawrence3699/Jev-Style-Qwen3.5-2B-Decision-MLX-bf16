# GitHub mirror of `Jev-Style-Qwen3.5-2B-Decision-MLX-bf16`

This repository mirrors the public Hugging Face model at:

<https://huggingface.co/chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16>

Snapshot of Hugging Face revision `63d20ba55ad024cf60b6a169a467183adf8057ba`.

Files smaller than 100 MiB are stored on the `main` branch. Larger files are attached to this GitHub Release:

<https://github.com/lawrence3699/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16/releases/tag/huggingface-snapshot-2026-09-23>

See `RELEASE_ASSETS.tsv` for asset names, original paths, sizes, and SHA-256 checksums. Asset names use `__` in place of `/` for files stored in subfolders.

## Reassembling split files

Assets ending in `.part-aa`, `.part-ab`, and so on are consecutive pieces of one original file. Download every piece and concatenate them in lexical order. For example:

```bash
cat model.safetensors.part-* > model.safetensors
```

Verify the reconstructed file against the `original_sha256` value in `RELEASE_ASSETS.tsv`.
