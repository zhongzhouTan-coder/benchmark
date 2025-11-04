import ast

from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.exceptions import AISBenchDataContentError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES

logger = AISLogger()


class DataProcessor:

    def __init__(self):
        pass

    def read_data(self, data):
        logger.info(f"Starting data processing for {len(data)} items")
        for idx, item in enumerate(data):
            logger.debug(f"Processing item {idx+1}: key_answer_type={item.get('key_answer_type')}")
            if isinstance(item['standard_answer_range'],
                          str) and item['key_answer_type'] != 'math':
                try:
                    item['standard_answer_range'] = ast.literal_eval(
                        item['standard_answer_range'])
                    logger.debug(f"Evaluated standard_answer_range for item {idx+1}")
                except Exception as e:
                    raise AISBenchDataContentError(UTILS_CODES.INVALID_TYPE, f"Invalid standard_answer_range format in item {idx+1}: {e}") from e

            item['standard_answer_range'] = str(item['standard_answer_range'])
            item['key_answer_type'] = str(item['key_answer_type'])

        logger.info(f"Data processing completed for {len(data)} items")
        return data
