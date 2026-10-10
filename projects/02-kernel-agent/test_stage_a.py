"""Regression tests for the ten-level Stage A challenge deliverable."""

import types
import unittest

import numpy as np

import kernelbench as kb
import stage_a_agent as agent


def load_kernel(source, name="candidate"):
    namespace = {}
    exec(compile(source, name, "exec"), namespace)
    return namespace["kernel"]


class StageAHarnessTests(unittest.TestCase):
    def test_all_ten_levels_have_explicit_hostile_cases(self):
        self.assertEqual(list(kb.LEVELS), list(range(1, 11)))
        for level in (1, 2, 3, 4, 5, 6, 8):
            cases = kb.LEVELS[level]["build"]()
            labels = {label for label, _ in cases}
            self.assertIn("strictly negative", labels)
            self.assertIn("constant rows", labels)
            self.assertIn("large mean tiny variance", labels)
            self.assertTrue(any(1 in np.shape(args[0]) for _, args in cases))
            self.assertTrue(any(127 in np.shape(args[0]) or 257 in np.shape(args[0])
                                for _, args in cases))

        attention_cases = kb.LEVELS[9]["build"]()
        self.assertTrue(any(args[3] > args[0].shape[0] for _, args in attention_cases))
        convolution_cases = kb.LEVELS[10]["build"]()
        self.assertTrue(any(args[3] > 1 for _, args in convolution_cases))

    def test_known_good_kernels_are_rule_clean_and_numerically_verified(self):
        for level, source in agent.OFFLINE_GOOD.items():
            with self.subTest(level=level):
                self.assertEqual(kb.check_rules(source, level), [])
                result = kb.verify(load_kernel(source, f"good-{level}"), level)
                self.assertTrue(result["ok"], result["failures"][:1])

    def test_each_planted_bug_is_caught_and_classified(self):
        expected = {
            1: "ragged_edge",
            2: "ragged_edge",
            3: "max_identity",
            4: "accumulator_precision",
            5: "softmax_stability",
            6: "ragged_edge",
            7: "matmul_k_tail",
            8: "layernorm_cancellation",
            9: "attention_window",
            10: "convolution_geometry",
        }
        for level, source in kb.PLANTED_BUGS.items():
            with self.subTest(level=level), np.errstate(over="ignore", invalid="ignore",
                                                        divide="ignore"):
                result = kb.verify(load_kernel(source, f"bad-{level}"), level)
                self.assertFalse(result["ok"])
                verdict = "\n".join(message for _, message in result["failures"])
                self.assertEqual(agent.failure_category(verdict), expected[level])
                self.assertNotEqual(agent.instruction_from_verdict(verdict).strip(), "")

    def test_static_checker_rejects_common_shortcuts_and_oversized_tiles(self):
        no_loop = "import numpy as np\ndef kernel(x, a, b):\n    return np.maximum(a*x+b, 0)\n"
        self.assertTrue(any("no explicit loops" in item
                            for item in kb.check_rules(no_loop, 1)))

        oversized = """def kernel(x):
    rows, cols = x.shape
    for r0 in range(0, rows, 256):
        tile = x[r0:r0+256, 0:512]
    return x
"""
        violations = kb.check_rules(oversized, 2)
        self.assertTrue(any("exceeds maximum 128" in item for item in violations))

        transpose = "def kernel(x):\n    for i in range(1):\n        return x.T\n"
        self.assertTrue(any("whole transpose" in item
                            for item in kb.check_rules(transpose, 6)))

        matmul = "def kernel(a, b):\n    for i in range(1):\n        return a @ b\n"
        self.assertTrue(any("banned matrix multiply" in item
                            for item in kb.check_rules(matmul, 7)))

    def test_summary_contains_required_challenge_artifacts(self):
        args = types.SimpleNamespace(offline=True, model="unused", context=8192,
                                     log="attempts.jsonl")
        history = {
            1: [{"ok": True, "rounds": 2, "confidence": 1.0,
                 "category": "verified", "token_usage": {
                     "attempts": 2, "prompt_tokens": 0,
                     "completion_tokens": 100,
                     "input_reference": 20,
                 }}]
        }
        summary = agent.build_summary(args, [1], history,
                                      {"ragged_edge": 1, "verified": 1})
        self.assertTrue(summary["all_verified"])
        self.assertEqual(summary["failure_taxonomy"], {"ragged_edge": 1})
        self.assertIn("token_instrumentation", summary)
        self.assertEqual(summary["tolerance"]["rtol"], kb.DEFAULT_RTOL)


if __name__ == "__main__":
    unittest.main()
