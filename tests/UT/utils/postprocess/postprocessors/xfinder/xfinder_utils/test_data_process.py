import pytest

from ais_bench.benchmark.utils.postprocess.postprocessors.xfinder.xfinder_utils.data_process import (
    DataProcessor,
)
from ais_bench.benchmark.utils.logging.exceptions import AISBenchDataContentError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES


def test_read_data_processes_valid_alphabet_option():
    processor = DataProcessor()
    data = [{
        'standard_answer_range': "['A', 'B']",
        'key_answer_type': 'alphabet_option'
    }]
    result = processor.read_data(data)
    assert result[0]['standard_answer_range'] == "['A', 'B']"
    assert result[0]['key_answer_type'] == 'alphabet_option'


def test_read_data_evaluates_string_standard_answer_range():
    processor = DataProcessor()
    data = [{
        'standard_answer_range': "['option1', 'option2']",
        'key_answer_type': 'alphabet_option'
    }]
    result = processor.read_data(data)
    # After eval and then str
    assert result[0]['standard_answer_range'] == "['option1', 'option2']"
    assert isinstance(result[0]['standard_answer_range'], str)


def test_read_data_skips_eval_for_math_type():
    processor = DataProcessor()
    data = [{
        'standard_answer_range': "['not_evaled']",
        'key_answer_type': 'math'
    }]
    result = processor.read_data(data)
    # Should not eval, just str
    assert result[0]['standard_answer_range'] == "['not_evaled']"


def test_read_data_raises_on_invalid_standard_answer_range():
    processor = DataProcessor()
    data = [{
        'standard_answer_range': "invalid[",
        'key_answer_type': 'alphabet_option'
    }]
    with pytest.raises(AISBenchDataContentError) as exc_info:
        processor.read_data(data)
    assert exc_info.value.error_code_str == UTILS_CODES.INVALID_TYPE.full_code


def test_read_data_converts_fields_to_strings():
    processor = DataProcessor()
    data = [{
        'standard_answer_range': ['A', 'B'],  # list
        'key_answer_type': 123  # int
    }]
    result = processor.read_data(data)
    assert result[0]['standard_answer_range'] == "['A', 'B']"
    assert result[0]['key_answer_type'] == '123'


def test_read_data_handles_multiple_items():
    processor = DataProcessor()
    data = [
        {
            'standard_answer_range': "['A']",
            'key_answer_type': 'alphabet_option'
        },
        {
            'standard_answer_range': "12345",
            'key_answer_type': 'math'
        }
    ]
    result = processor.read_data(data)
    assert len(result) == 2
    assert result[0]['standard_answer_range'] == "['A']"
    assert result[1]['standard_answer_range'] == "12345"


def test_read_data_with_non_string_standard_answer_range():
    processor = DataProcessor()
    data = [{
        'standard_answer_range': ['A', 'B'],  # already list
        'key_answer_type': 'alphabet_option'
    }]
    result = processor.read_data(data)
    # Should not eval, just str
    assert result[0]['standard_answer_range'] == "['A', 'B']"