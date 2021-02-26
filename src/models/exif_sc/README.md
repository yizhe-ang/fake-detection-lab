# Evaluation Results
There are likely to be some hidden hyperparameter choices that are not replicated exactly in these experiments.

In the paper, every dataset in split in half into train / test sets. Since the exact split is unknown, all my experiments simply evaluate on the entire dataset.

## Splice Detection
Average precision scores are reported.

|          | Results from Paper | This Implementation |
| -------- | ------------------ | ------------------- |
| RT       | 0.55               | 0.54                |
| Columbia | 0.98               | 0.95                |

## Splice Localization
Class-balanced IOU (cIOU) scores are reported.

I resize all the ground-truth and prediction maps into the same size in order to compute the optimal threshold and corresponding IoU score for each image in a vectorized and efficient manner.

|             | Results from Paper | This Implementation |
| ----------- | ------------------ | ------------------- |
| RT          | 0.54               | 0.54                |
| Columbia    | 0.85               | 0.67                |
| In-the-Wild | 0.58               | 0.64                |
| Hays        | 0.65               | 0.54                |

## Qualitative Results
### Realistic Tampering
![](/assets/exif_sc_examples/realistic_tampering.png)

### Columbia
![](/assets/exif_sc_examples/columbia.png)

### In-the-Wild
![](/assets/exif_sc_examples/in_the_wild.png)

### Scene Completion (Hays)
![](/assets/exif_sc_examples/scene_completion.png)