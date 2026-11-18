from __future__ import annotations

import unittest

from src.integrity import IntegrityError
from src.model_worker import classify_execution_error
from src.provenance import sha256_text
from src.prompt import SYSTEM_PROMPT


class ErrorClassificationTests(unittest.TestCase):
    def test_error_categories_are_explicit(self):
        self.assertEqual(classify_execution_error(KeyboardInterrupt()), "recoverable_interruption")
        self.assertEqual(classify_execution_error(IntegrityError("bad grid")), "integrity_failure")
        self.assertEqual(classify_execution_error(RuntimeError("CUDA out of memory")), "cuda_oom")
        self.assertEqual(
            classify_execution_error(RuntimeError("other")), "fatal_model_execution_failure"
        )

    def test_prompt_hash_is_stable_sha256(self):
        digest = sha256_text(SYSTEM_PROMPT)
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, sha256_text(SYSTEM_PROMPT))


if __name__ == "__main__":
    unittest.main()
