from ais_bench.benchmark.utils.postprocess import model_postprocessors as mp


class DummyPool:
    def __init__(self, n):
        self.n = n
        self._batches = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def map(self, func, batches):
        # Capture batches for assertions and run synchronously
        self._batches.extend(list(batches))
        return [func(batch) for batch in self._batches]


def iter_identity(x):
    return x


def test_gen_output_naive_mutates_and_collects(monkeypatch):
    monkeypatch.setattr(mp, "tqdm", iter_identity)

    class DummyExtractor:
        def prepare_input(self, item):
            return f"{item['q']}|{item['o']}"

        def gen_output(self, user_input):
            return f"E:{user_input}"

    items = [
        {"q": "Q1", "o": "O1"},
        {"q": "Q2", "o": "O2"},
        {"q": "Q3", "o": "O3"},
    ]

    out = mp.gen_output_naive(items, DummyExtractor())
    assert out == ["E:Q1|O1", "E:Q2|O2", "E:Q3|O3"]
    # Each original item should be mutated with extracted_answer
    for i, it in enumerate(items, 1):
        assert it["extracted_answer"] == f"E:Q{i}|O{i}"


def test_gen_output_xfinder_mutates_and_collects(monkeypatch):
    monkeypatch.setattr(mp, "tqdm", iter_identity)

    class DummyExtractor:
        def prepare_input(self, item):
            return f"{item['question']}|{item['llm_output']}"

        def gen_output(self, user_input):
            return f"X:{user_input}"

    items = [
        {
            "key_answer_type": "t",
            "standard_answer_range": "r",
            "correct_answer": "g1",
            "question": "Q1",
            "llm_output": "O1",
        },
        {
            "key_answer_type": "t",
            "standard_answer_range": "r",
            "correct_answer": "g2",
            "question": "Q2",
            "llm_output": "O2",
        },
    ]

    answers, pairs, data = mp.gen_output_xfinder(items, DummyExtractor())
    assert answers == ["X:Q1|O1", "X:Q2|O2"]
    # ext_cor_pairs structure
    assert pairs == [["t", "r", "X:Q1|O1", "g1"], ["t", "r", "X:Q2|O2", "g2"]]
    # items mutated with xfinder_extracted_answer
    for i, it in enumerate(items, 1):
        assert it["xfinder_extracted_answer"] == f"X:Q{i}|O{i}"
    # data mirrors input order and includes mutations
    assert data == items


def test_naive_model_postprocess_batching(monkeypatch):
    # Replace Pool with dummy to capture batches; skip tqdm overhead
    dp = DummyPool(0)
    monkeypatch.setattr(mp, "Pool", lambda n: dp)
    monkeypatch.setattr(mp, "tqdm", iter_identity)

    # formatter returns 5 items
    def fake_format(_preds):
        return [{"q": f"Q{i}", "o": f"O{i}"} for i in range(1, 6)]

    monkeypatch.setattr(mp, "format_input_naive", fake_format)

    # Replace gen_output_naive with a shim that returns one marker per item
    batch_sizes = []

    def shim_gen_output_naive(ori_data, **kwargs):  # noqa: ARG001
        batch_sizes.append(len(ori_data))
        return ["E" for _ in ori_data]

    monkeypatch.setattr(mp, "gen_output_naive", shim_gen_output_naive)

    class DummyExtractor:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(mp, "NaiveExtractor", DummyExtractor)

    # With num_processes=2 and 5 items, batch_size = 5//2 = 2 -> batches: 2,2,1
    answers = mp.naive_model_postprocess(
        preds=[{}] * 5, model_name="m", custom_instruction="", api_url="http://u", num_processes=2
    )

    assert len(answers) == 5
    assert batch_sizes == [2, 2, 1]


def test_xfinder_postprocess_batching(monkeypatch):
    dp = DummyPool(0)
    monkeypatch.setattr(mp, "Pool", lambda n: dp)
    monkeypatch.setattr(mp, "tqdm", iter_identity)

    monkeypatch.setattr(mp, "convert_to_xfinder_format", lambda qt, preds: [
        {"seed": i} for i in range(5)
    ])

    class DummyDataProcessor:
        def read_data(self, texts):
            # Provide minimal fields used by gen_output_xfinder
            out = []
            for i in range(len(texts)):
                out.append({
                    "key_answer_type": "t",
                    "standard_answer_range": "r",
                    "correct_answer": f"g{i}",
                    "question": f"Q{i}",
                    "llm_output": f"O{i}",
                })
            return out

    class DummyExtractor:
        def __init__(self, model_name, url):
            pass

        def prepare_input(self, item):
            return f"{item['question']}|{item['llm_output']}"

        def gen_output(self, user_input):
            return f"X:{user_input}"

    monkeypatch.setattr(mp, "DataProcessor", DummyDataProcessor)
    monkeypatch.setattr(mp, "Extractor", DummyExtractor)

    # Replace gen_output_xfinder to record batch sizes
    batch_sizes = []

    def shim_gen_output_xfinder(ori_data, **kwargs):  # noqa: ARG001
        batch_sizes.append(len(ori_data))
        # returns (answers, pairs, data)
        answers = ["X" for _ in ori_data]
        pairs = [["t", "r", "X", "g"] for _ in ori_data]
        return answers, pairs, ori_data

    monkeypatch.setattr(mp, "gen_output_xfinder", shim_gen_output_xfinder)

    answers = mp.xfinder_postprocess(
        preds=[{}] * 5, question_type="qt", model_name="m", api_url="http://u", 
    )

    assert len(answers) == 5
    # By default num_processes computed inside helper to <= len, batch_size = len//num = 5//5 = 1
    # but since we didn't pass num_processes into xfinder _eval_pred, it uses default 8 then clamps to len
    # so batches should be five 1-sized batches
    assert batch_sizes == [1, 1, 1, 1, 1]


def test_extract_non_reasoning_content_only_end_token():
    t = "This is a test.</think> How are you?"
    assert mp.extract_non_reasoning_content(t) == "How are you?"


def test_extract_non_reasoning_content_both_tokens():
    t = "Start<think>reasoning here</think> End"
    assert mp.extract_non_reasoning_content(t) == "Start End"


def test_extract_non_reasoning_content_no_tokens():
    t = "Plain text"
    assert mp.extract_non_reasoning_content(t) == "Plain text"


def test_extract_non_reasoning_content_list_input():
    inputs = [
        "Start<think>reasoning</think> End",
        "Test</think> Result",
    ]
    assert mp.extract_non_reasoning_content(inputs) == ["Start End", "Result"]


def test_extract_non_reasoning_content_custom_tokens():
    single = "A <r>reason</r> B"
    assert (
        mp.extract_non_reasoning_content(
            single, think_start_token="<r>", think_end_token="</r>"
        )
        == "A  B".strip()
    )

    lst = ["P <r>a</r> Q", "X <r>b</r> Y"]
    assert mp.extract_non_reasoning_content(
        lst, think_start_token="<r>", think_end_token="</r>"
    ) == ["P  Q".strip(), "X  Y".strip()]


def test_extract_non_reasoning_content_multiple_pairs():
    t = "A<think>x</think>B<think>y</think>C"
    assert mp.extract_non_reasoning_content(t) == "ABC"


def test_extract_non_reasoning_content_only_start_token():
    t = "Hello <think>unfinished"
    assert mp.extract_non_reasoning_content(t) == t.strip()
