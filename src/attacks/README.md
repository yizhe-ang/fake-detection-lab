# Evaluation Results
Localization metrics are reported.

Only the spliced images in the datasets were used for evaluation. For the MCC metric that requires binary decision maps instead of score maps, a default threshold value of 0.5 was chosen (most papers search for the optimal threshold value and report the corresponding highest metric score instead). The F1 score was computed by finding the optimal threshold.

For the details of the attack, a step size of 10000 was used, with 50 iterations.

| Dataset          | F1 ↑   | MCC ↑   | mAP ↑  | AUC ↑  |    cIoU ↑ |
| ---------------- | ------ | ------- | ------ | ------ | ------ |
| Columbia         | 0.8703 | 0.6971  | 0.8958 | 0.9697 | 0.8490 |
| **Adv-Columbia** | 0.7014 | 0.0004  | 0.6984 | 0.8773 | 0.7194 |
| DSO-1            | 0.9473 | 0.3650  | 0.9652 | 0.8439 | 0.5038 |
| **Adv-DSO-1**    | 0.9263 | 0.02211 | 0.9313 | 0.7303 | 0.5263 |

# Qualitative Results

## Columbia Dataset
![](/assets/lots_examples/columbia_1.png)

![](/assets/lots_examples/columbia_2.png)

## DSO-1 Dataset
![](/assets/lots_examples/dso1_1.png)

![](/assets/lots_examples/dso1_2.png)
