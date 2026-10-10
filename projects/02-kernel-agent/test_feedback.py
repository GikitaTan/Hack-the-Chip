"""CPU-only regression tests for the structured checker feedback."""

import json
import unittest

from feedback import Feedback, classify_error
import agent


class FeedbackClassificationTests(unittest.TestCase):
    def test_known_errors_have_stable_categories(self):
        cases = {
            "No code came back.": "EMPTY_CODE",
            "The code does not parse: invalid syntax on line 2.": "SYNTAX_ERROR",
            "dma_copy dst partition dimension 256 exceeds maximum 128": "PARTITION_LIMIT",
            "dma_copy requires src and dst to have the same number of elements, got src=64, dst=128": "DMA_SIZE_MISMATCH",
            "value array of shape (128,) could not be broadcast to indexing result of shape (64,)": "TILE_SHAPE_MISMATCH",
            "foo() got an unexpected keyword argument 'axis'": "INVALID_API_ARGUMENT",
            "'MemoryRegion' object is not callable": "MEMORY_REGION_MISUSE",
            "foo() got multiple values for argument 'dst'": "DUPLICATE_API_ARGUMENT",
            "module 'nki.language' has no attribute 'scalar'": "UNKNOWN_API",
            "No module named 'nki.nl'": "UNKNOWN_MODULE",
            "unsupported operand type(s) for +=: 'NkiTensor' and 'NkiTensor'": "TENSOR_PYTHON_OPERATOR",
            "Out-of-bound access for tensor x on dimension 1: index range [0, 512] exceed dimension size of 500": "OUT_OF_BOUNDS",
            "Matmul contraction dimension 256 exceeds pmax=128": "MATMUL_CONTRACTION_LIMIT",
            "tile must have at least 2 dimensions": "TILE_RANK",
            "cannot reshape array of size 64 into shape (8,9)": "INVALID_RESHAPE",
            "moving must be in ['sbuf'], got shared_hbm": "BUFFER_PLACEMENT",
            "no function named `kernel` is defined": "MISSING_ENTRYPOINT",
            "`kernel` is not decorated with `@nki.jit`": "MISSING_JIT_DECORATOR",
            "WRONG SHAPE: returned (4, 3), reference is (3, 4).": "WRONG_OUTPUT_SHAPE",
            "NON-FINITE OUTPUT: 1 NaN and 0 Inf, first at (0, 0).": "NON_FINITE_OUTPUT",
            "THE KERNEL MODIFIED ITS INPUT (argument 0).": "INPUT_MUTATION",
            "OUTPUT IS 100% ZEROS while the reference is not.": "OUTPUT_NOT_WRITTEN",
            "40% of the output is zero, in the block from (0, 0) to (3, 3).": "PARTIAL_OUTPUT",
            "NUMERICAL MISMATCH: worst error 1.0. this is in the final partial PARTITION tile -- the ragged edge is the likely cause": "RAGGED_EDGE_MISMATCH",
            "CORRECT ON CPU BUT WRONG ON HARDWARE: unsafe access": "HARDWARE_HAZARD",
            "CORRECT, BUT TOO MUCH HBM TRAFFIC FOR THIS LEVEL: moving 2.50x the byte floor, and level 5 requires 1.50x or better.": "HBM_TRAFFIC_LIMIT",
            "CANNOT SIMULATE: the Neuron SDK is not installed here": "SDK_UNAVAILABLE",
            "The file imports but kernel could not be loaded: ValueError": "KERNEL_LOAD_ERROR",
            "raised RuntimeError: device failure": "RUNTIME_EXCEPTION",
            "Correct on every shape.": "VERIFIED",
        }
        for raw, expected in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(classify_error(raw).category, expected)

    def test_partition_limit_extracts_actionable_fields(self):
        feedback = classify_error(
            "dma_copy dst partition dimension 256 exceeds maximum 128",
            stage="runtime", level=4, case="M=256 K=128")
        self.assertEqual(feedback.family, "TILE_LAYOUT")
        self.assertEqual(feedback.severity, "structural")
        self.assertEqual(feedback.current, 256)
        self.assertEqual(feedback.maximum, 128)
        self.assertTrue(feedback.redesign_required)
        self.assertIn("at most 128", feedback.suggestion)

    def test_unknown_error_is_preserved_and_json_serialisable(self):
        raw = "an entirely new SDK diagnostic"
        feedback = classify_error(raw)
        self.assertEqual(feedback.category, "UNKNOWN_ERROR")
        self.assertEqual(feedback.raw_error, raw)
        json.dumps(feedback.to_dict())

    def test_enrich_is_idempotently_classifiable(self):
        raw = "dma_copy dst partition dimension 256 exceeds maximum 128"
        rendered = agent.enrich(raw)
        self.assertEqual(classify_error(rendered).category, "PARTITION_LIMIT")

    def test_grade_keeps_legacy_tuple_and_detailed_api(self):
        reward, parts, prompt = agent.grade("", 1)
        self.assertEqual(reward, 0.0)
        self.assertFalse(parts["parses"])
        self.assertIsInstance(prompt, str)
        self.assertIn("EMPTY_CODE", prompt)

        reward2, parts2, feedback = agent.grade_detailed("", 1)
        self.assertEqual((reward2, parts2), (reward, parts))
        self.assertIsInstance(feedback, Feedback)
        self.assertEqual(feedback.category, "EMPTY_CODE")

    def test_static_rule_reward_is_unchanged(self):
        source = "def wrong_name(x):\n    return x\n"
        reward, parts, feedback = agent.grade_detailed(source, 1)
        self.assertEqual(reward, agent.WEIGHTS["parses"])
        self.assertTrue(parts["parses"])
        self.assertFalse(parts["rules"])
        self.assertEqual(feedback.family, "RULE")


if __name__ == "__main__":
    unittest.main()
