import torch

from comfy.ldm.modules.attention import attention_pytorch


def test_causal_sdpa_matches_dense_causal_mask():
    torch.manual_seed(1)
    q = torch.randn(1, 4, 19, 8)
    k = torch.randn(1, 2, 19, 8)
    v = torch.randn(1, 2, 19, 8)
    mask = torch.full((19, 19), torch.finfo(q.dtype).min / 4).triu_(1)

    dense = attention_pytorch(q, k, v, 4, mask=mask, skip_reshape=True, enable_gqa=True)
    causal = attention_pytorch(q, k, v, 4, skip_reshape=True, enable_gqa=True, is_causal=True)

    torch.testing.assert_close(causal, dense, rtol=1e-5, atol=1e-6)
