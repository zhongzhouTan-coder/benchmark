from ais_bench.benchmark.registry import DICT_POSTPROCESSORS
from ais_bench.benchmark.utils.logging import AISLogger


logger = AISLogger()


@DICT_POSTPROCESSORS.register_module('base')
def base_postprocess(output: dict) -> dict:
    logger.debug(
        f"base_postprocess: passthrough dict with keys={list(output.keys()) if isinstance(output, dict) else 'n/a'}"
    )
    return output
