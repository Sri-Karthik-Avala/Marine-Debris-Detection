# Marine Debris Detection

| | |
| --- | --- |
| Final rank | #4 |
| Domain | Object Detection |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | CPU |
| Challenge status | Accepted / closed |
| Creator | jacekwladzinski |
| Solutions submitted | 4 |
| Last submission | 2026-09-15 |

## Problem statement

### Overview

A coastal survey image has been partly reviewed: some annotated debris boxes are already in a ledger, while other annotated objects still need entries. Detect the remaining objects in the same image and localize their annotated extents precisely so each new entry provides a close object crop. Use both the pixels and the supplied ledger, and do not return boxes that are already logged.

All debris is one detection class. Examples include floats, rope, netting, manufactured fragments, tires, processed wood, and stranded vessels. Natural rocks, vegetation, water, and unmodified driftwood are not automatically debris. Training boxes illustrate the annotation scope. Fine-grained material identification, removal priority, object mass, and physical access are not requested.

Every supplied ledger box is an exact human annotation from that row's image. The target is the complement of those ledger annotations for the review state; it does not require deciding whether two views show the same physical object. Every released row has at least one target box. The ledger can be empty, and its length does not determine how many targets remain.

The images contain small objects, dense accumulations, edge-truncated objects, and black regions without image coverage. Targets follow the provided human box annotations, including annotated truncated objects. No additional minimum object size applies. There are no instance masks or fine-grained category targets. An unannotated region is not independently certified debris-free; scores measure agreement with the annotated catalogue, not exhaustive environmental ground truth. Visually plausible but unmatched proposals receive no match credit and count toward the prediction total.

Use CPU only, at most 62GB of RAM, and at most 90 minutes for the complete solution, including preprocessing, training, validation, inference, and submission writing.

### Dataset

There are 1,010 training rows and 411 test rows, with one distinct 640 × 640 RGB JPEG per row under `images/`. Each row is one image plus one partial-review state. The same ground can appear in several source images. `survey_id` identifies a geographic dependency group suitable for grouped validation: training and test group IDs are disjoint, and known overlapping views and shared orthophoto sources stay together. These groups are not certified independent flights or physical specimens. Keep all rows sharing a `survey_id` together during validation.

The ledger is image-local. It is not a tracking history, change-over-time signal, or cross-view physical-object match. Public inputs contain everything required to interpret it. Do not search for source annotations or infer test labels from unavailable external records.

`train.csv`

| Column | Type | Meaning |

|---|---|---|

| `id` | String | Unique review-state ID. |

| `image` | Relative path | JPEG input under `images/`. |

| `survey_id` | String | Geographic group for validation and score aggregation. |

| `logged_boxes` | JSON array | Debris boxes already annotated in this image. |

| `new_boxes` | JSON array | Remaining annotated debris boxes to predict. |

Both box columns use arrays of `[left,top,right,bottom]` values. Coordinates are measured in pixels from the image's upper-left corner, with x increasing rightward and y downward. Boxes use continuous half-open extents `[left,right) × [top,bottom)` inside the image. Coordinates lie in `[0,640]`; width and height must be positive. Ledger boxes can touch an image edge. You may combine `logged_boxes` and `new_boxes` to train a detector for all annotated debris, then remove detector proposals matching the row's ledger. For example, suppressing proposals at IoU at least 0.5 to a logged box is a permitted heuristic; the 0.81 threshold in evaluation applies to matching remaining targets, not to a mandatory ledger-suppression algorithm.

`test.csv`

The columns are `id,image,survey_id,logged_boxes`, with the same meanings as in training. Generate one prediction row for every test ID, including rows for which your model predicts no remaining objects.

`sample_submission.csv`

The columns are `id,new_boxes`. Every `new_boxes` value is `[]`; this is a valid abstention example, not a competitive detector. It uses no hidden object counts. Image files, CSV inputs, and this sample are the complete model-input package.

### Submission format

Write `working/submission.csv` as a UTF-8 CSV with the required columns `id` and `new_boxes`. Rows and columns may be reordered. Additional uniquely named columns are ignored consistently. Column names must be nonempty strings without surrounding spaces. Duplicate headers, blank or ragged rows, malformed CSV quoting, missing IDs, repeated IDs, or extra IDs invalidate the entire submission. The CSV may start with a UTF-8 byte-order mark.

For these invented examples, image `demo_a` contains three annotated objects at `[100,100,140,140]`, `[300,200,350,250]`, and `[500,400,540,460]`; its ledger contains the first and third. Image `demo_b` contains objects at `[0,20,25,65]` and `[200,300,230,335]`; its ledger contains the second. The two submission rows below return only the remaining boxes, including an edge-touching target.

```
id,new_boxes

demo_a,"[[300,200,350,250]]"

demo_b,"[[0,20,25,65]]"
```

### Evaluation

For boxes A and B, `IoU(A,B) = area(A ∩ B) / area(A ∪ B)`, using their continuous pixel areas. A prediction can match a target in the same row if IoU is at least **0.81**. This requires close agreement with the annotation: a box that merely surrounds the correct object can miss the localization requirement. Matching maximizes the number of qualifying one-to-one matches. A prediction and a target can each participate in at most one match. Equality at IoU 0.81 counts as a match. For example, `[0,0,81,100]` against `[0,0,100,100]` qualifies exactly, whereas `[0,0,80,100]` does not. Multiple maximum matchings have the same match count and therefore the same score; enumeration and tie order do not affect scoring.

Within each geographic group in the scoring cohort, sum the matched pairs `TP`, the number of submitted boxes `P`, and the number of target boxes `T` over its rows. The group score is `2 × TP / (P + T)` when `P + T > 0`. If both totals are zero, the group score is 1. Returning no boxes for a group with targets scores 0. Unmatched boxes, including duplicate proposals and already logged objects, increase `P` without increasing `TP`; missing objects increase `T` without increasing `TP`.

The final score is the arithmetic mean of the group scores, a float from 0 to 1 to maximize. Each group present in the supplied scoring cohort has equal weight. If a cohort contains only part of a group, its totals use those supplied rows only. There is no confidence ranking, score power, or hidden cutoff. Correct complete predictions attain 1. A full-size image rectangle is not a substitute for localized objects.

### Not allowed

-Remote inference services

- External task images, task labels, models trained on this debris collection, source-to-answer lookup, and manual annotation of test images are not allowed.
- Training, selecting thresholds, or changing hyperparameters using test predictions, test distribution statistics, or test labels is not allowed. Use only training-side validation groups for those decisions.
- Do not transfer labels between test images or update model weights at test time. Each output must depend only on its row's image and ledger plus a model fitted on training data.
