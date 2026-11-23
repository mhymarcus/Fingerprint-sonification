import unittest

import cv2
import numpy as np

from utils.orientation import estimate_orientation_field


class OrientationFieldTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.ones((96, 96), dtype=np.uint8)

    def test_horizontal_ridges_produce_horizontal_orientation(self):
        image = np.full((96, 96), 255, dtype=np.uint8)
        for row in range(8, 96, 10):
            cv2.line(image, (0, row), (95, row), 0, 2)

        angles, coherence = estimate_orientation_field(
            image, self.mask, block_size=16, smooth=False
        )

        axial_distance = np.minimum(angles, np.pi - angles)
        self.assertLess(float(np.nanmedian(axial_distance)), 0.05)
        self.assertGreater(float(np.median(coherence)), 0.9)

    def test_vertical_ridges_produce_vertical_orientation(self):
        image = np.full((96, 96), 255, dtype=np.uint8)
        for column in range(8, 96, 10):
            cv2.line(image, (column, 0), (column, 95), 0, 2)

        angles, coherence = estimate_orientation_field(
            image, self.mask, block_size=16, smooth=False
        )

        self.assertAlmostEqual(float(np.nanmedian(angles)), np.pi / 2, places=4)
        self.assertGreater(float(np.median(coherence)), 0.9)


if __name__ == "__main__":
    unittest.main()
