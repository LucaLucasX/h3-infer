# MiniMax H3 TE residency and DiT prefetch experiment

Production tree: `/mnt/luca/H3_infer/ComfyUI-master_cp` (not modified)

Experiment tree: `/mnt/luca/H3_infer/ComfyUI-master_cp_te_exp`

## Runtime switches

- `COMFY_H3_VISION_RESIDENT=1` enables the managed 1.109 GiB BF16 vision replica.
- `COMFY_H3_VISION_RESIDENT=0` disables the replica and uses normal dynamic VBAR loading.
- The experimental DiT lookahead implementation was removed after A/B testing.

## 9-image 704-short-edge, 1280x704, 124-frame, Sage+Sparge+4-step A/B

Baseline cold DiT steps: 21.300 / 7.684 / 8.177 / 8.205 seconds.

Lookahead, no cross-step, cold DiT steps: 9.387 / 7.723 / 7.804 / 7.859 seconds.

Baseline warm DiT steps: 7.765 / 7.786 / 7.843 / 7.923 seconds (31.317 total).

Lookahead, no cross-step, warm DiT steps: 7.747 / 7.868 / 8.080 / 7.874 seconds (31.569 total).

Decoded video-frame and audio-frame hashes matched exactly for seeds 42 and 43.

Cross-step retention was rejected: AIMDO reported pinned VBAR pages and denoising output diverged after step zero.

The DiT lookahead source was subsequently removed from the experiment tree.

## 9-image + 3x5-second-reference-video benchmark

Output: 1280x704, 124 frames, Sage+Sparge+LightX2V four-step, seed 44.

- Production baseline: TE 28.467 seconds, DiT 110.886 seconds, total 214.80 seconds.
- Optimized TE: TE 9.332 seconds, DiT 108.807 seconds, total 200.73 seconds.
- TE saving: 19.135 seconds (67.2 percent), 3.05x throughput.
- End-to-end saving: 14.07 seconds (6.55 percent), 1.07x throughput.
- Under DiT load, the managed vision cache was pressure-evicted successfully (`evicted=1.109GiB`).

## Rollback

No production files were changed. Stop the experiment service and continue using the production tree.
Within the experiment tree, vision residency can be disabled at runtime with
`COMFY_H3_VISION_RESIDENT=0`.
