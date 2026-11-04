from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np

from pigeon.backends.tflite import TFLiteEngine
from pigeon.models.yolo import yolo26


class TestCPURegressions(unittest.TestCase):
    def test_bird_classifier_preserves_quantized_rgb(self):
        engine = TFLiteEngine.__new__(TFLiteEngine)
        engine.interpreter = Mock()
        engine.input_details = [{
            "index": 0, "shape": [1, 1, 3, 3], "dtype": np.uint8,
            "quantization": (1 / 128, 128),
        }]
        engine.output_details = [{"index": 1}]
        engine.interpreter.get_tensor.return_value = np.array([[0.9]], dtype=np.float32)
        pixels = np.array([[[0, 128, 255], [255, 0, 128], [128, 255, 0]]], dtype=np.uint8)
        engine.classify(pixels)
        np.testing.assert_array_equal(engine.interpreter.set_tensor.call_args.args[1], pixels[None])

    def test_yolo_uint8_boxes_and_duplicate_suppression(self):
        engine = Mock()
        engine.get_input_details.return_value = [
            {"index": 0, "dtype": np.uint8, "quantization": (1 / 255, 0)}
        ]
        engine.get_output_details.return_value = [
            {"index": 1, "shape": [1, 4, 2], "quantization": (0.01, 0)},
            {"index": 2, "shape": [1, 80, 2], "quantization": (0.01, 0)},
        ]
        boxes = np.array([[[50, 50], [50, 50], [20, 20], [40, 40]]], dtype=np.int8)
        scores = np.zeros((1, 80, 2), dtype=np.int8)
        scores[0, 14] = [90, 80]
        engine.get_tensor.side_effect = [boxes, scores]
        objects = yolo26(np.full((100, 100, 3), 255, np.uint8), engine, 0.5, 0.5, (0, 10))
        self.assertEqual(engine.set_tensor.call_args.args[1].dtype, np.uint8)
        self.assertEqual(int(engine.set_tensor.call_args.args[1].min()), 255)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].id, 15)
        np.testing.assert_allclose(objects[0].bbox, [80, 40, 120, 120])
