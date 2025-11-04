import pytest

from ais_bench.benchmark.utils.postprocess.postprocessors.xfinder.xfinder_utils.convert_data import (
    parse_options,
    convert_to_xfinder_format,
    xfinder_template,
)
from ais_bench.benchmark.utils.logging.exceptions import ParameterValueError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES


def test_parse_options_with_standard_formats():
    text = """(A) First option
(B) Second option
(C) Third option"""
    result = parse_options(text)
    assert result == [['A', 'First option'], ['B', 'Second option'], ['C', 'Third option']]


def test_parse_options_with_varied_formats():
    text = """A) Option A
B. Option B
C: Option C
(D) Option D"""
    result = parse_options(text)
    assert result == [['A', 'Option A'], ['B', 'Option B'], ['C', 'Option C'], ['D', 'Option D']]


def test_parse_options_empty_text():
    result = parse_options("")
    assert result == []


def test_parse_options_no_matches():
    text = "This is just plain text without options."
    result = parse_options(text)
    assert result == []


def test_parse_options_mixed_content():
    text = """Some intro text.
(A) First choice
Irrelevant line
(B) Second choice
More text."""
    result = parse_options(text)
    assert result == [['A', 'First choice'], ['B', 'Second choice']]


def test_convert_to_xfinder_format_invalid_type():
    with pytest.raises(ParameterValueError) as exc_info:
        convert_to_xfinder_format('invalid_type', [])
    assert exc_info.value.error_code_str == UTILS_CODES.INVALID_TYPE.full_code


def test_convert_to_xfinder_format_alphabet_option():
    data = [{
        'origin_prompt': [{'role': 'HUMAN', 'prompt': 'Question?\n(A) Yes\n(B) No'}],
        'prediction': 'A',
        'reference': 'A',
        'gold': 'A'
    }]
    result = convert_to_xfinder_format('alphabet_option', data, 'TestModel', 'TestDataset')
    assert len(result) == 1
    item = result[0]
    assert item['key_answer_type'] == 'alphabet_option'
    assert item['question'] == 'Question?\n(A) Yes\n(B) No'
    assert item['llm_output'] == 'A'
    assert item['correct_answer'] == 'A'
    assert item['model_name'] == 'TestModel'
    assert item['dataset'] == 'TestDataset'
    assert item['standard_answer_range'] == [['A', 'Yes'], ['B', 'No']]


def test_convert_to_xfinder_format_short_text():
    data = [{
        'origin_prompt': [{'role': 'HUMAN', 'prompt': 'What is AI?'}],
        'prediction': 'Artificial Intelligence',
        'reference': None,
        'gold': 'AI stands for Artificial Intelligence'
    }]
    result = convert_to_xfinder_format('short_text', data)
    assert len(result) == 1
    item = result[0]
    assert item['key_answer_type'] == 'short_text'
    assert item['standard_answer_range'] == 'AI stands for Artificial Intelligence'


def test_convert_to_xfinder_format_categorical_label():
    data = [{
        'origin_prompt': [{'role': 'HUMAN', 'prompt': 'Category?'}],
        'prediction': 'Type A',
        'reference': 'Type A',
        'gold': 'Type A'
    }]
    result = convert_to_xfinder_format('categorical_label', data)
    assert len(result) == 1
    item = result[0]
    assert item['key_answer_type'] == ''
    assert item['standard_answer_range'] == []


def test_convert_to_xfinder_format_math():
    data = [{
        'origin_prompt': [{'role': 'HUMAN', 'prompt': '2+2=?'}],
        'prediction': '4',
        'reference': '4',
        'gold': '4'
    }]
    result = convert_to_xfinder_format('math', data)
    assert len(result) == 1
    item = result[0]
    assert item['key_answer_type'] == 'math'
    assert item['standard_answer_range'] == 'a(n) number / set / vector / matrix / interval / expression / function / equation / inequality'


def test_convert_to_xfinder_format_with_parsing_error():
    data = [{
        'origin_prompt': [],  # Empty list to cause IndexError on [-1]
        'prediction': 'A',
        'reference': 'A',
        'gold': 'A'
    }]
    result = convert_to_xfinder_format('alphabet_option', data)
    # Should skip the item due to parsing error
    assert len(result) == 0


def test_convert_to_xfinder_format_multiple_items():
    data = [
        {
            'origin_prompt': [{'role': 'HUMAN', 'prompt': 'Q1?\n(A) A\n(B) B'}],
            'prediction': 'A',
            'reference': 'A',
            'gold': 'A'
        },
        {
            'origin_prompt': [{'role': 'HUMAN', 'prompt': 'Q2?'}],
            'prediction': 'Answer',
            'reference': None,
            'gold': 'Answer'
        }
    ]
    result = convert_to_xfinder_format('alphabet_option', data)
    assert len(result) == 2
    assert result[0]['question'] == 'Q1?\n(A) A\n(B) B'
    assert result[1]['question'] == 'Q2?'


def test_xfinder_template_structure():
    # Ensure all expected keys are present
    expected_types = ['math', 'alphabet_option', 'categorical_label', 'short_text']
    for typ in expected_types:
        assert typ in xfinder_template
        template = xfinder_template[typ]
        required_keys = ['model_name', 'dataset', 'key_answer_type', 'question', 'llm_output', 'correct_answer', 'standard_answer_range']
        for key in required_keys:
            assert key in template