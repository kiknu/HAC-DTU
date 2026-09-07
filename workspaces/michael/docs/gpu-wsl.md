# CUDA under WSL

The validated stack is Torch 2.6.0+cu124 with the pinned PyTorch3D revision.
The Windows NVIDIA driver supplies WSL's `libcuda`; do not install a separate
Linux display driver inside WSL.

Check the complete path with:

```bash
nvidia-smi
uv run hac-asteroids doctor
```

If `nvidia-smi` reports that GPU access is blocked and `/dev/dxg` is absent,
repeat the check outside the current container or tool sandbox. In the
September 2026 workspace diagnosis, the RTX 3080 and CUDA tensor allocation
worked normally outside the Codex sandbox; no WSL or driver repair was needed.
