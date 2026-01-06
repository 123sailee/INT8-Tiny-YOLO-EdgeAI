Dataset instructions

This folder should contain information about dataset structure and how
to prepare a COCO-format detection dataset for experiments.

Recommended dataset layout (COCO):

- data/
  - images/
    - train/
    - val/
    - test/
  - annotations/
    - instances_train.json
    - instances_val.json

Notes:
- The evaluation scripts in `evaluation/` expect COCO-format annotations.
- For quick testing, use a small subset of COCO or a custom dataset converted to COCO.

References:
- COCO format: http://cocodataset.org
