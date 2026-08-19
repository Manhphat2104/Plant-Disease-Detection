import sys
import importlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


class FakeProcessor:
    def __call__(self, images, return_tensors):
        self.last_image = images
        self.last_return_tensors = return_tensors
        return {"pixel_values": torch.zeros(1, 3, 224, 224)}


class LoadImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = FakeProcessor()
        processor_patch = patch(
            "transformers.ViTImageProcessor.from_pretrained",
            return_value=cls.processor,
        )
        processor_patch.start()
        cls.addClassCleanup(processor_patch.stop)

        cls.inference = importlib.import_module("inference")

    def assert_valid_result(self, image_input):
        pil_image, pixel_values = self.inference.load_image(image_input)

        self.assertIsInstance(pil_image, Image.Image)
        self.assertEqual(pil_image.mode, "RGB")
        self.assertEqual(pixel_values.shape, (1, 3, 224, 224))
        self.assertEqual(self.processor.last_return_tensors, "pt")

    def test_loads_image_from_file_path(self):
        with TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "leaf.png"
            Image.new("L", (32, 32), color=128).save(image_path)

            self.assert_valid_result(str(image_path))

    def test_loads_image_from_numpy_array(self):
        image_array = np.zeros((32, 32, 3), dtype=np.uint8)

        self.assert_valid_result(image_array)

    def test_loads_image_from_pil_image(self):
        image = Image.new("RGBA", (32, 32), color=(10, 20, 30, 255))

        self.assert_valid_result(image)


if __name__ == "__main__":
    unittest.main()