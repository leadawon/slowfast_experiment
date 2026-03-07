from model import LLaDA,Dream
from sampling_utils import   set_seed
import os
import sys
from lm_eval.__main__ import cli_evaluate


def _extract_output_path(argv):
    for i, arg in enumerate(argv):
        if arg.startswith("--output_path="):
            return arg.split("=", 1)[1]
        if arg == "--output_path" and i + 1 < len(argv):
            return argv[i + 1]
    return None


if __name__ == "__main__":
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
    output_path = _extract_output_path(sys.argv[1:])
    if output_path and "LM_EVAL_OUTPUT_PATH" not in os.environ:
        os.environ["LM_EVAL_OUTPUT_PATH"] = output_path
    set_seed(1234)
    cli_evaluate()
