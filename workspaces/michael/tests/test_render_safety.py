"""Renderer resource-limit regression tests."""

import unittest

from render_safety import (
    enforce_face_limit,
    estimate_raster_memory,
    refuse_public_raw_stl,
)


class RenderSafetyTests(unittest.TestCase):
    def test_face_limit_requires_explicit_override(self):
        enforce_face_limit(5_000)
        with self.assertRaisesRegex(ValueError, "5,001"):
            enforce_face_limit(5_001)
        enforce_face_limit(5_001, allow_large_mesh=True)

    def test_raster_estimate_scales_with_all_dimensions(self):
        base = estimate_raster_memory(1, 64, 4, requires_grad=True)
        self.assertEqual(
            estimate_raster_memory(2, 64, 4, requires_grad=True).bytes,
            2 * base.bytes,
        )
        self.assertGreater(
            estimate_raster_memory(1, 128, 4, requires_grad=True).bytes,
            base.bytes,
        )
        self.assertGreater(
            estimate_raster_memory(1, 64, 8, requires_grad=True).bytes,
            base.bytes,
        )
        self.assertLess(
            estimate_raster_memory(1, 64, 4, requires_grad=False).bytes,
            base.bytes,
        )

    def test_public_raw_stl_is_refused(self):
        with self.assertRaisesRegex(ValueError, "raw public STL"):
            refuse_public_raw_stl("challenge_data/asteroid1.stl")
        refuse_public_raw_stl("proxy.obj")

    def test_staged_renderer_settings_are_encoded(self):
        from pytorch3d_forward import RendererConfig
        self.assertEqual(
            (RendererConfig.smoke().image_size, RendererConfig.smoke().face_limit),
            (32, 80),
        )
        self.assertEqual(
            (RendererConfig.smoke().max_frames, RendererConfig.smoke().max_cameras),
            (1, 1),
        )
        self.assertEqual(
            (RendererConfig.gradient_test().image_size, RendererConfig.gradient_test().face_limit),
            (64, 320),
        )
        self.assertEqual(
            (RendererConfig.gradient_test().max_frames,
             RendererConfig.gradient_test().max_cameras),
            (2, 1),
        )
        self.assertEqual(
            (RendererConfig.optimization().image_size, RendererConfig.optimization().face_limit),
            (96, 1280),
        )
        self.assertTrue(RendererConfig.evaluation().require_no_grad)


if __name__ == "__main__":
    unittest.main()
