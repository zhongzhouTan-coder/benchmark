# Convert AISBench prediction data to XFinder format
import copy
import re

from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.exceptions import ParameterValueError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES

logger = AISLogger()

xfinder_template = {
    'math': {
        'model_name':
        '',
        'dataset':
        '',
        'key_answer_type':
        'math',
        'question':
        '',
        'llm_output':
        '',
        'correct_answer':
        '',
        'standard_answer_range':
        'a(n) number / set / vector / matrix / interval / expression / function / equation / inequality'  # noqa
    },
    'alphabet_option': {
        'model_name': '',
        'dataset': '',
        'key_answer_type': 'alphabet_option',
        'question': '',
        'llm_output': '.',
        'correct_answer': '',
        'standard_answer_range': []
    },
    'categorical_label': {
        'model_name': '',
        'dataset': '',
        'key_answer_type': '',
        'question': '',
        'llm_output': '',
        'correct_answer': '',
        'standard_answer_range': []
    },
    'short_text': {
        'model_name': '',
        'dataset': '',
        'key_answer_type': 'short_text',
        'question': '',
        'llm_output': '',
        'correct_answer': '',
        'standard_answer_range': []
    }
}


def parse_options(text: str):
    lines = text.split('\n')
    parsed_options = []
    option_pattern = r'^[A-Z]\)|[A-Z]\.|[A-Z]\)|[A-Z]:|\([A-Z]\)'
    for line in lines:
        line = line.strip()
        match = re.match(option_pattern, line)
        if match:
            option = ''
            # 等于第一个属于选项的字符
            for c in line:
                if c.isalpha():
                    option = c
                    break
            content_start = match.end() + 1
            content = line[content_start:].strip()
            parsed_options.append([option, content])

    return parsed_options


def convert_to_xfinder_format(typ, data, model_name='', dataset_name=''):
    if typ not in xfinder_template.keys():
        raise ParameterValueError(UTILS_CODES.INVALID_TYPE, f'Invalid type {typ}')
    logger.info(f"Starting conversion to xFinder format: type={typ}, data_items={len(data)}, model_name={model_name}, dataset_name={dataset_name}")
    format_data = []
    skipped_count = 0
    for idx, item in enumerate(data):
        template = copy.deepcopy(xfinder_template[typ])
        try:
            question = item['origin_prompt'][-1]['prompt']
            logger.debug(f"Processing item {idx+1}/{len(data)}: question_preview={question[:50]}...")
            llm_output = item['prediction']
            correct_answer = item['reference'] if item['reference'] else item[
                'gold']
            template['correct_answer'] = correct_answer
            template['model_name'] = model_name
            template['dataset'] = dataset_name
            template['question'] = question
            template['llm_output'] = llm_output
            if typ == 'alphabet_option':
                options = parse_options(question)
                template['standard_answer_range'] = options
                logger.debug(f"Parsed {len(options)} options for alphabet_option")
            elif typ == 'short_text':
                template['standard_answer_range'] = item['gold']
            elif typ == 'categorical_label':
                pass
        except (ValueError, KeyError, TypeError, AttributeError, IndexError) as e:
            logger.warning(f'Error when parsing question options for item {idx+1}: {type(e).__name__}: {e}, skipping item')
            skipped_count += 1
            continue

        format_data.append(template)
    logger.info(f"Conversion completed: successfully_converted={len(format_data)}, skipped={skipped_count}, total={len(data)}")
    return format_data
