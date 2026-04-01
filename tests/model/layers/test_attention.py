"""Tests for attention layers to verify numerical equivalence after optimization."""

import unittest

import torch
from pytorch_lightning import seed_everything

from boltz.model.layers.attention import AttentionPairBias


class AttentionPairBiasTest(unittest.TestCase):
    """Test AttentionPairBias layer."""

    C_S = 64
    C_Z = 32
    NUM_HEADS = 4
    BATCH = 2
    SEQ_Q = 16
    SEQ_K = 16

    def setUp(self):
        seed_everything(42)
        torch.set_grad_enabled(False)

        self.layer = AttentionPairBias(
            c_s=self.C_S,
            c_z=self.C_Z,
            num_heads=self.NUM_HEADS,
        )
        self.layer.eval()

    def _make_inputs(self, seq_q=None, seq_k=None):
        """Create random inputs for the attention layer."""
        sq = seq_q or self.SEQ_Q
        sk = seq_k or self.SEQ_K
        s = torch.randn(self.BATCH, sq, self.C_S)
        z = torch.randn(self.BATCH, sq, sk, self.C_Z)
        mask = torch.ones(self.BATCH, sk)
        return s, z, mask

    def test_basic_forward(self):
        """Test that forward pass produces correct output shape."""
        s, z, mask = self._make_inputs()
        out = self.layer(s, z, mask)
        self.assertEqual(out.shape, s.shape)

    def test_masked_positions_ignored(self):
        """Test that masking a position changes the output."""
        # Use non-zero initialized weights so masking has visible effect
        seed_everything(123)
        layer = AttentionPairBias(
            c_s=self.C_S, c_z=self.C_Z, num_heads=self.NUM_HEADS
        )
        # Initialize proj_o with non-zero weights
        torch.nn.init.normal_(layer.proj_o.weight, std=0.1)
        layer.eval()

        s, z, mask = self._make_inputs()

        out_full = layer(s, z, mask)

        mask_partial = mask.clone()
        mask_partial[:, -1] = 0
        out_partial = layer(s, z, mask_partial)

        # Outputs should differ when masking changes attention distribution
        max_diff = (out_full - out_partial).abs().max().item()
        self.assertGreater(max_diff, 1e-6)

    def test_multiplicity(self):
        """Test that multiplicity correctly repeats pair bias."""
        s_single = torch.randn(1, self.SEQ_Q, self.C_S)
        z_single = torch.randn(1, self.SEQ_Q, self.SEQ_K, self.C_Z)
        mask_single = torch.ones(1, self.SEQ_K)

        # Run with multiplicity=1
        out1 = self.layer(s_single, z_single, mask_single, multiplicity=1)
        self.assertEqual(out1.shape, (1, self.SEQ_Q, self.C_S))

        # Run with multiplicity=2 (s is batched as 2, z is 1 and gets repeated)
        s_double = s_single.repeat(2, 1, 1)
        mask_double = mask_single.repeat(2, 1)
        out2 = self.layer(s_double, z_single, mask_double, multiplicity=2)
        self.assertEqual(out2.shape, (2, self.SEQ_Q, self.C_S))

        # Both samples should be identical since inputs are the same
        torch.testing.assert_close(out2[0], out2[1], atol=1e-5, rtol=1e-5)

    def test_deterministic(self):
        """Test that eval mode produces deterministic outputs."""
        s, z, mask = self._make_inputs()
        out1 = self.layer(s, z, mask)
        out2 = self.layer(s, z, mask)
        torch.testing.assert_close(out1, out2, atol=0, rtol=0)

    def test_model_cache(self):
        """Test that model cache produces identical results."""
        s, z, mask = self._make_inputs()

        # Without cache
        out_no_cache = self.layer(s, z, mask)

        # With cache (first call fills it)
        cache = {}
        out_cached_1 = self.layer(s, z, mask, model_cache=cache)
        self.assertIn("z", cache)

        # With cache (second call uses it)
        out_cached_2 = self.layer(s, z, mask, model_cache=cache)

        torch.testing.assert_close(out_no_cache, out_cached_1, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(out_cached_1, out_cached_2, atol=0, rtol=0)

    def test_gradient_flow(self):
        """Test that gradients flow through the layer."""
        torch.set_grad_enabled(True)
        # Use layer with non-zero proj_o for gradient flow
        seed_everything(99)
        layer = AttentionPairBias(
            c_s=self.C_S, c_z=self.C_Z, num_heads=self.NUM_HEADS
        )
        torch.nn.init.normal_(layer.proj_o.weight, std=0.1)
        layer.train()
        s, z, mask = self._make_inputs()
        s.requires_grad_(True)
        out = layer(s, z, mask)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(s.grad)
        self.assertTrue(s.grad.abs().sum() > 0)
        torch.set_grad_enabled(False)


class AttentionPairBiasV2Test(unittest.TestCase):
    """Test AttentionPairBias v2 layer."""

    C_S = 64
    C_Z = 32
    NUM_HEADS = 4
    BATCH = 2
    SEQ_Q = 16
    SEQ_K = 12

    def setUp(self):
        seed_everything(42)
        torch.set_grad_enabled(False)

        from boltz.model.layers.attentionv2 import AttentionPairBias as AttentionV2

        self.layer = AttentionV2(
            c_s=self.C_S,
            c_z=self.C_Z,
            num_heads=self.NUM_HEADS,
        )
        self.layer.eval()

    def test_basic_forward(self):
        """Test that forward pass produces correct output shape."""
        s = torch.randn(self.BATCH, self.SEQ_Q, self.C_S)
        z = torch.randn(self.BATCH, self.SEQ_Q, self.SEQ_K, self.C_Z)
        mask = torch.ones(self.BATCH, self.SEQ_K)
        k_in = torch.randn(self.BATCH, self.SEQ_K, self.C_S)
        out = self.layer(s, z, mask, k_in)
        self.assertEqual(out.shape, s.shape)

    def test_deterministic(self):
        """Test that eval mode produces deterministic outputs."""
        s = torch.randn(self.BATCH, self.SEQ_Q, self.C_S)
        z = torch.randn(self.BATCH, self.SEQ_Q, self.SEQ_K, self.C_Z)
        mask = torch.ones(self.BATCH, self.SEQ_K)
        k_in = torch.randn(self.BATCH, self.SEQ_K, self.C_S)
        out1 = self.layer(s, z, mask, k_in)
        out2 = self.layer(s, z, mask, k_in)
        torch.testing.assert_close(out1, out2, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
