# Evaluation Results
Localization metrics are reported.

Only the spliced images in the datasets were used for evaluation. For the MCC metric that requires binary decision maps instead of score maps, a default threshold value of 0.5 was chosen (most papers search for the optimal threshold value and report the corresponding highest metric score instead). The F1 score was computed by finding the optimal threshold.

Types of attack:
- **AdvMean**: sets all target features to be the mean feature of all authentic patches.
- **AdvSample**: samples uniformly from the set of features of authentic patches to be the target features for non-authentic patches.

A step size of 10000 was used, with 50 iterations.

| Dataset            | F1 ↑       | MCC ↑      | mAP ↑      | AUC ↑      | cIoU ↑     |
| ------------------ | ---------- | ---------- | ---------- | ---------- | ---------- |
| Columbia           | 0.8703     | 0.6971     | 0.8958     | 0.9697     | 0.8490     |
| AdvMean-Columbia   | 0.7014     | **0.0004** | 0.6984     | 0.8773     | 0.7194     |
| JPEG-Columbia      | 0.6397     | 0.2417     | 0.6084     | 0.8476     | 0.6528     |
| AdvSample-Columbia | **0.5067** | 0.0081     | **0.3832** | **0.7213** | **0.5363** |
|                    |
| DSO-1              | 0.9473     | 0.3650     | 0.9652     | 0.8439     | 0.5038     |
| AdvMean-DSO-1      | 0.9263     | **0.0221** | 0.9313     | 0.7303     | 0.5263     |
| JPEG-DSO-1         | **0.9253** | 0.1209     | 0.9195     | **0.6774** | 0.5124     |
| AdvSample-DSO-1    | 0.9281     | 0.0541     | **0.9129** | 0.6877     | **0.5041** |

# Qualitative Results

## Columbia Dataset

![](/assets/lots_examples/columbia_1.png)

![](/assets/lots_examples/columbia_2.png)

## DSO-1 Dataset

![](/assets/lots_examples/dso_1.png)

![](/assets/lots_examples/dso_2.png)
