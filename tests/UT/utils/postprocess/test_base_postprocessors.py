import unittest
import importlib
from unittest.mock import patch

from ais_bench.benchmark.utils.postprocess.base_postprocessors import base_postprocess


# Import module to patch module-level logger reliably
bp_module = importlib.import_module(
    "ais_bench.benchmark.utils.postprocess.base_postprocessors"
)


class TestBasePostprocessors(unittest.TestCase):
    def test_base_postprocess_passthrough_and_logs_keys(self):
        payload = {"a": 1, "b": 2}
        with patch.object(bp_module, "logger") as mock_logger:
            out = base_postprocess(payload)

        # Passthrough contract
        self.assertIs(out, payload)
        # Logging call occurred
        mock_logger.debug.assert_called_once()
        msg = mock_logger.debug.call_args[0][0]
        # Keys appear in the log message
        self.assertIn("a", msg)
        self.assertIn("b", msg)

    def test_base_postprocess_with_non_dict_logs_na(self):
        payload = [1, 2, 3]
        with patch.object(bp_module, "logger") as mock_logger:
            out = base_postprocess(payload)  # function returns input unchanged

        self.assertIs(out, payload)
        mock_logger.debug.assert_called_once()
        msg = mock_logger.debug.call_args[0][0]
        self.assertIn("n/a", msg)


if __name__ == "__main__":
    unittest.main()
