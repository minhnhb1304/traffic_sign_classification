import unittest

import numpy as np

from app.realtime.overlay import compute_center_roi, draw_overlay


class OverlayTest(unittest.TestCase):
    def test_compute_center_roi(self):
        self.assertEqual(compute_center_roi(480, 640, 0.5), (200, 120, 240))

    def test_draw_overlay_supports_vietnamese_label(self):
        img = np.zeros((120, 180, 3), dtype=np.uint8)

        out = draw_overlay(
            img,
            (20, 50, 60),
            "Giới hạn tốc độ",
            0.923,
            detected=True,
        )

        self.assertEqual(out.shape, img.shape)
        self.assertFalse(np.array_equal(out, img))


if __name__ == "__main__":
    unittest.main()