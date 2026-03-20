# RefCOCOg

[中文](README.md) | English

## Dataset Introduction

RefCOCOg is another variant in the RefCOCO family, mainly used to evaluate object localization under longer and more complex referring expressions. AISBench currently provides two split tasks, `val` and `test`, with about 7.57k and 5.02k raw samples respectively.

In AISBench, RefCOCOg reuses the same loading and evaluation flow as RefCOCO. The dataset is stored as parquet shards, with image bytes embedded directly in the parquet files. The loader expands each row's `answer` list into multiple benchmark samples and supports both the default file-path mode with runtime caches under `data/temp_save_images/<split>/` and an opt-in base64 mode for API-based multimodal calls. Predictions are evaluated with `BBoxIoUEvaluator` using `Accuracy@0.5` for Intersection over Union (IoU) $\ge 0.5$. In the current configs, `coord_scale=1000.0` matches a common `0-1000` normalized bounding box convention, but this is model-specific rather than a universal requirement.

- Task type: MMU 2D grounding (multimodal visual grounding / referring expression comprehension)
- Dataset characteristic: compared with RefCOCO, RefCOCOg usually contains longer and more linguistically complex descriptions
- Data format: `<split>-*.parquet` files under the `data/` directory

> 🔗 Dataset Homepage [https://huggingface.co/datasets/lmms-lab/RefCOCOg](https://huggingface.co/datasets/lmms-lab/RefCOCOg)

## Dataset Deployment

- The dataset source is the HuggingFace dataset repository `lmms-lab/RefCOCOg`.
- The config reads from `{tool_root_path}/ais_bench/datasets/RefCOCOg/data/` by default, so the full repository should be downloaded into `{tool_root_path}/ais_bench/datasets/RefCOCOg/`.
- On Linux, you can deploy it with the following commands:

```bash
# On a Linux server, under the tool root path
cd ais_bench/datasets
mkdir -p RefCOCOg
huggingface-cli download lmms-lab/RefCOCOg --repo-type dataset --local-dir RefCOCOg
```

- Run `tree RefCOCOg/` under `{tool_root_path}/ais_bench/datasets`. If the directory structure looks like the following, the dataset has been deployed correctly:

```text
RefCOCOg/
├── .gitattributes
├── README.md
└── data/
    ├── test-*.parquet
    └── val-*.parquet
```

- When using the default file-path config, the loader will automatically create JPEG caches under `data/temp_save_images/<split>/` on the first evaluation run. You do not need to create that directory in advance.

## Available Dataset Tasks

| Task Name           | Introduction                                                                                                                                                                           | Evaluation Metric | Few-Shot | Prompt Format                             | Config File                                      |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | -------- | ----------------------------------------- | ------------------------------------------------ |
| refcocog_gen        | RefCOCOg generative grounding config that uses file-path image input (`file://{image}`) and exports `RefCOCOg_val` and `RefCOCOg_test` split tasks                                     | Accuracy@0.5      | 0-shot   | Multimodal chat format (MMPromptTemplate) | [refcocog_gen.py](refcocog_gen.py)               |
| refcocog_gen_base64 | RefCOCOg generative grounding config that uses base64 data-URL image input (`data:image/jpeg;base64,{image}`) and exports `RefCOCOg_base64_val` and `RefCOCOg_base64_test` split tasks | Accuracy@0.5      | 0-shot   | Multimodal chat format (MMPromptTemplate) | [refcocog_gen_base64.py](refcocog_gen_base64.py) |

## Dataset Classification

Use `--datasets` with the following dataset config filenames:

- `refcocog_gen`: file-path image input config, defined in [refcocog_gen.py](refcocog_gen.py).
- `refcocog_gen_base64`: base64 image input config, defined in [refcocog_gen_base64.py](refcocog_gen_base64.py).

At runtime, `refcocog_gen` exports two split dataset abbreviations: `RefCOCOg_val` and `RefCOCOg_test`. `refcocog_gen_base64` exports `RefCOCOg_base64_val` and `RefCOCOg_base64_test`. These abbreviations are mainly used in outputs, logs, and summarizer displays, while both configs share the same reader and evaluator logic.

## Usage Examples

The following minimal config snippet highlights the parameters that matter most in RefCOCOg dataset tasks.

File-path mode example:

```python
from ais_bench.benchmark.openicl.icl_prompt_template import MMPromptTemplate

refcocog_reader_cfg = dict(
    input_columns=['question', 'image'],
    output_column='answer',
)

refcocog_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt_mm={
                    'text': {
                        'type': 'text',
                        'text': 'Locate every object that matches the description "{question}" in the image. Report bbox coordinates in JSON format.'
                    },
                    'image': {'type': 'image_url', 'image_url': {'url': 'file://{image}'}},
                })
            ]
        )
    ),
)

dataset = dict(
    abbr='RefCOCOg_val',
    path='ais_bench/datasets/RefCOCOg/data',
    split='val',
    reader_cfg=refcocog_reader_cfg,
    infer_cfg=refcocog_infer_cfg,
)

refcocog_eval_cfg = dict(
    evaluator=dict(
        type=BBoxIoUEvaluator,
        iou_threshold=0.5,
        coord_scale=1000.0,
        smart_resize_cfg=dict(
            factor=32,
            min_pixels=65536,
            max_pixels=16 * 16 * 4 * 16384,
        ),
    ),
    pred_postprocessor=dict(type=refcoco_bbox_postprocess),
)
```

- `input_columns=['question', 'image']`: `format_mm` fills `{question}` and `{image}` in the multimodal prompt template from the dataset sample.
- `output_column='answer'`: the evaluator reads the reference bounding box payload from the dataset's `answer` column.
- `image_url.url='file://{image}'`: the prompt passes a local image path to the model.
- `abbr`: the dataset abbreviation used in outputs, logs, and summary tables.
- `split='val'`: this dataset entry loads parquet shards matching `val-*.parquet`.
- `iou_threshold=0.5`: a prediction is counted as correct only when the Intersection over Union (IoU) between the predicted box and the reference box is at least `0.5`.
- `coord_scale=1000.0`: in the current example config, model outputs are interpreted on a `0-1000` normalized bounding box scale. This is a model-specific parameter; the current value is mainly used to align with the 2D grounding coordinate convention described in the Qwen3-VL technical report, and should be changed if your model uses a different bounding box scale.
- `smart_resize_cfg`: before receiving an image, the model applies a smart resize that matches the Qwen3-VL image preprocessor, scaling the image to satisfy the `factor`, `min_pixels`, and `max_pixels` constraints. The evaluator uses this config to map the model's bounding box output — which is expressed in resized-image coordinate space — back to the original image coordinate space before computing Intersection over Union (IoU) against the ground-truth box.
- `pred_postprocessor=refcoco_bbox_postprocess`: AISBench first extracts the first bounding box coordinate sequence from the model output text before evaluation.

Base64 mode differs from the file-path config in two key places:

```python
dataset = dict(
    abbr='RefCOCOg_base64_val',
    image_type='base64',
)

image_field = {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,{image}'}}
```

- `image_type='base64'`: the loader encodes images from parquet into base64 strings instead of materializing temporary JPEG files.
- `data:image/jpeg;base64,{image}`: the prompt embeds the image bytes directly as a data URL, which is suitable for API-based multimodal models.

## Usage Recommendations

- Because RefCOCOg expressions are longer, use a multimodal grounding model that remains stable when generating bbox-only answers and does not add too much unrelated text.
- `refcoco_bbox_postprocess` only extracts the first four-value coordinate sequence from the prediction text, so JSON array output or a JSON object with a `bbox` field is recommended.
- `coord_scale=1000.0` is the model-alignment setting used by the current RefCOCOg configs. It is suitable for models that emit `0-1000` normalized bounding box coordinates; if your model emits original-image pixels or another scale, adjust this parameter accordingly.
- [refcocog_gen.py](refcocog_gen.py) uses file-path image input, and its prompt image field is formatted as `file://{image}`. This is the recommended default mode.
- [refcocog_gen_base64.py](refcocog_gen_base64.py) uses base64 data-URL image input, and its prompt image field is formatted as `data:image/jpeg;base64,{image}`. Use it for API-based multimodal models that require base64 data URLs.
