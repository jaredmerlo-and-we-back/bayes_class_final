# Bayesian Intrusion Detection with CIC-IDS2017

**Course:** Bayesian Machine Learning
**Dataset:** [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) — Canadian Institute for Cybersecurity

---

## Abstract

Modern intrusion detection systems (IDS) rely heavily on rule-based signatures that fail to generalize to novel attack patterns. This project applies **Bayesian probabilistic modeling** to the CIC-IDS2017 network traffic dataset to classify flows as benign or malicious.

We implement and compare two models:

1. **Gaussian Naive Bayes (from scratch)** — a generative Bayesian classifier built without any ML libraries, using Bayes' theorem with Gaussian class-conditional likelihoods.
2. **Bayesian Logistic Regression (PyMC + ADVI)** — a discriminative model with Normal(0,1) priors on all coefficients, fit via Automatic Differentiation Variational Inference (ADVI), yielding full approximate posterior distributions over model weights.

The dataset contains ~2.8 million labeled network flow records spanning one week of simulated traffic, including eight attack categories (DDoS, port scanning, web attacks, infiltration, and more).

---

## Table of Contents

1. [Dataset](#dataset)
2. [Preprocessing](#preprocessing)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Feature Selection](#feature-selection)
5. [Gaussian Naive Bayes](#gaussian-naive-bayes)
6. [Model Evaluation — GNB](#model-evaluation--gnb)
7. [Bayesian Logistic Regression](#bayesian-logistic-regression)
8. [Results Summary](#results-summary)
9. [Conclusions](#conclusions)
10. [References](#references)

---

## Dataset

| File | Day | Traffic Types |
|---|---|---|
| `Monday-WorkingHours.pcap_ISCX.csv` | Monday | Benign only |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | Tuesday | FTP-Patator, SSH-Patator |
| `Wednesday-workingHours.pcap_ISCX.csv` | Wednesday | DoS (Hulk, GoldenEye, Slowloris, SlowHTTPTest) |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | Thursday AM | SQL Injection, XSS, Brute Force |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | Thursday PM | Infiltration |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | Friday AM | Botnet |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | Friday PM | Port Scan |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | Friday PM | DDoS |

After concatenation across all eight files, the dataset has **2,830,743 rows** and **80 columns**.

**Label distribution (raw):**

| Label | Count |
|---|---|
| BENIGN | 2,273,097 |
| DoS Hulk | 231,073 |
| PortScan | 158,930 |
| DDoS | 128,027 |
| DoS GoldenEye | 10,293 |
| FTP-Patator | 7,938 |
| SSH-Patator | 5,897 |
| DoS slowloris | 5,796 |
| DoS Slowhttptest | 5,499 |
| Bot | 1,966 |
| Web Attack — Brute Force | 1,507 |
| Web Attack — XSS | 652 |
| Infiltration | 36 |
| Web Attack — SQL Injection | 21 |
| Heartbleed | 11 |

> **Note:** The CSV files are not tracked in this repository due to their size (~1 GB total). Download from the [CIC website](https://www.unb.ca/cic/datasets/ids-2017.html) and place them in `MachineLearningCSV/MachineLearningCVE/`.

---

## Preprocessing

Before modeling, several cleaning steps are applied:

1. **Column name normalization** — strip leading/trailing whitespace from column headers (a known quirk of CIC-IDS2017 CSV exports).
2. **Binary label encoding** — collapse the multi-class `Label` column into a binary target: `0 = BENIGN`, `1 = any attack`.
3. **Infinity handling** — features like `Flow Bytes/s` and `Flow Packets/s` can be infinite due to near-zero flow durations (4,376 infinite values found). These are replaced with `NaN`.
4. **Missing value imputation** — columns with >5% missing values are dropped (none qualified). Remaining NaN values are filled with the column median to preserve distributional shape.

After binarization, the class distribution is:

| Class | Count | Proportion |
|---|---|---|
| Benign (0) | 2,273,097 | 80.3% |
| Attack (1) | 557,646 | 19.7% |

### Class Distribution

![Binary Class Distribution](binary_class_distribution.png)

**Interpretation:** The dataset is moderately imbalanced, with roughly 4:1 benign-to-attack ratio. This is realistic — in production networks, the vast majority of traffic is legitimate. The imbalance means accuracy alone is a misleading metric; precision, recall, and ROC AUC must be reported and interpreted with this context in mind. A naive classifier that always predicts "benign" would achieve 80.3% accuracy while catching zero attacks.

---

## Exploratory Data Analysis

To identify the most informative features, 100,000 rows are sampled for efficiency and Pearson correlations are computed between every feature and the binary label.

**Top features by absolute correlation with the attack label:**

| Feature | Correlation |
|---|---|
| Bwd Packet Length Std | +0.504 |
| Bwd Packet Length Max | +0.486 |
| Bwd Packet Length Mean / Avg Bwd Segment Size | +0.476 |
| Packet Length Std | +0.464 |
| Packet Length Variance | +0.449 |
| Max Packet Length | +0.449 |
| Fwd IAT Std | +0.425 |
| Min Packet Length | −0.307 |
| Bwd Packet Length Min | −0.279 |

### Feature Distributions (KDE Plots)

The following plots compare the kernel density estimates of each feature separately for benign vs. attack flows. Good features show a clear separation between the two distributions.

#### Bwd Packet Length Std
![Distribution of Bwd Packet Length Std](dist_bwd_packet_length_std.png)

**Interpretation:** Attack flows show a pronounced heavy right tail in backward packet length standard deviation. Benign flows cluster tightly near zero, while many attack flows have high within-flow variability in backward packet sizes. This strong separation (r = 0.504) makes it the single most predictive feature. The pattern is consistent with attack traffic like DDoS, where servers send varied-size responses or malicious payloads differ in size.

---

#### Bwd Packet Length Max
![Distribution of Bwd Packet Length Max](dist_bwd_packet_length_std_length_max.png)

**Interpretation:** The maximum backward packet length similarly separates the classes. Attack flows often involve unusually large packets — for example, server responses to DoS probes or data exfiltration payloads. Benign flows tend to produce more uniformly sized packets within a session.

---

#### Bwd Packet Length Mean
![Distribution of Bwd Packet Length Mean](dist_bwd_packet_length_std_mean.png)

**Interpretation:** Attack flows have a bimodal distribution in mean backward packet length — a large spike at zero (many attack flows elicit no server response) and a secondary mode at higher values. Benign flows cluster at moderate, consistent response sizes. This bimodality reflects the diverse attack taxonomy in the dataset.

---

#### Avg Bwd Segment Size
![Distribution of Avg Bwd Segment Size](dist_bwd_packet_avg_bwd_segment_size.png)

**Interpretation:** Avg Bwd Segment Size is highly correlated with Bwd Packet Length Mean (r ≈ 1.0 between them), as both measure average data segment sizes on the return path. The distribution mirrors the above: attack flows cluster near zero or at extreme values, while benign flows have a tighter, higher central tendency. This confirms the signal is real and not an artifact of a single feature definition.

---

#### Packet Length Std
![Distribution of Packet Length Std](dist_packet_length_std.png)

**Interpretation:** Overall packet length variability (across both directions) is elevated in attack flows. Benign traffic tends to be more regular — consistent request/response sizes reflect human-driven web sessions, file transfers, or streaming protocols. Attack traffic like port scans or DDoS often mix tiny probe packets with large payloads, producing high within-flow variance.

---

#### Packet Length Variance
![Distribution of Packet Length Variance](dist_packet_length_variance.png)

**Interpretation:** Packet length variance tells the same story as the standard deviation (mathematically equivalent up to squaring). The right-skewed distribution for attacks confirms that high variance in packet sizes is a reliable attack indicator. Because variance and standard deviation carry near-identical information, including both in the model would be redundant — this is addressed in the correlation heatmap below.

---

### Correlation Heatmap

![Correlation Heatmap](correlation%20heatmap.png)

**Interpretation:** The heatmap reveals strong intercorrelations among the candidate features. Notably:

- `Packet Length Std` and `Packet Length Variance` are perfectly correlated (r = 1.0) — only one should be included.
- `Bwd Packet Length Mean` and `Avg Bwd Segment Size` are nearly identical (r ≈ 1.0).
- The Bwd Packet Length family (Std, Max, Mean) all correlate strongly with each other.

Including perfectly correlated features does not add information and inflates numerical issues (near-singular covariance in GNB). The final feature set is selected to be both predictive and non-redundant.

---

## Feature Selection

Based on EDA correlations and distribution plots, six features are selected that show strong and complementary discriminative power:

| Feature | Correlation (r) | Rationale |
|---|---|---|
| `Packet Length Std` | +0.464 | High within-flow packet size variability signals attack traffic |
| `Max Packet Length` | +0.449 | Attacks often involve unusually large individual packets |
| `Fwd IAT Std` | +0.425 | Irregular inter-arrival times indicate automated (non-human) flows |
| `Idle Mean` | +0.394 | Attack flows tend to have different idle behavior patterns |
| `Flow Duration` | varies | Short, high-volume flows are typical of DDoS and scan attacks |
| `Total Fwd Packets` | varies | Volume of forward packets distinguishes attack types |

All features are **standardized** (zero mean, unit variance) before fitting to satisfy the Gaussian likelihood assumption in GNB and to stabilize gradient estimation in ADVI.

**Train/test split:** 80/20 stratified split → 2,264,594 training rows, 566,149 test rows.

---

## Gaussian Naive Bayes

Naive Bayes is a generative Bayesian classifier applying Bayes' theorem under the conditional independence assumption: given the class label, each feature is independent of the others.

$$P(c \mid \mathbf{x}) \propto P(c) \prod_{i=1}^{d} P(x_i \mid c)$$

Each class-conditional likelihood is modeled as a Gaussian:

$$P(x_i \mid c) = \frac{1}{\sqrt{2\pi\sigma_{i,c}^2}} \exp\!\left(-\frac{(x_i - \mu_{i,c})^2}{2\sigma_{i,c}^2}\right)$$

**Implementation steps:**
1. Estimate class priors $P(c)$ from training label frequencies: $P(\text{benign}) = 0.803$, $P(\text{attack}) = 0.197$.
2. Compute per-class, per-feature means $\mu_{i,c}$ and variances $\sigma^2_{i,c}$ from training data.
3. At inference, compute log-posteriors for both classes and apply the **log-sum-exp trick** for numerical stability.
4. Threshold at 0.5 to produce hard predictions.

The model is implemented entirely from scratch in NumPy — no sklearn GaussianNB or similar library is used.

---

## Model Evaluation — GNB

### Quantitative Results

| Metric | Benign (0) | Attack (1) | Overall |
|---|---|---|---|
| Precision | 0.87 | 0.49 | — |
| Recall | 0.88 | 0.45 | — |
| F1 Score | 0.88 | 0.47 | — |
| Support | 454,620 | 111,529 | 566,149 |
| **Accuracy** | — | — | **0.80** |
| **ROC AUC** | — | — | **0.761** |

**Macro average:** Precision = 0.68, Recall = 0.67, F1 = 0.67
**Weighted average:** Precision = 0.79, Recall = 0.80, F1 = 0.80

### Confusion Matrix

![Confusion Matrix](confusion%20matrix.png)

**Interpretation:** The confusion matrix shows:
- **True Negatives (TN):** ~400,000 benign flows correctly classified as benign.
- **True Positives (TP):** ~50,000 attack flows correctly caught.
- **False Negatives (FN):** ~61,000 attack flows missed (classified as benign). These are the most dangerous errors in an IDS context — missed detections.
- **False Positives (FP):** ~55,000 benign flows flagged as attacks. These generate false alarms for network operators.

The 45% recall on the attack class means the GNB misses more than half of all attacks. Given the conditional independence assumption is almost certainly violated (packet length features are highly correlated), this is expected. The model is competitive but not production-ready without further feature engineering or a model that captures feature interactions.

---

### Posterior Probability of Attack

![Posterior Probability of Attack](posterior%20probability%20of%20attack.png)

**Interpretation:** This histogram shows the GNB's predicted probability of attack ($P(\text{attack} \mid \mathbf{x})$) separately for benign flows (blue) and true attack flows (orange). A well-calibrated, highly discriminative classifier would push the benign distribution to the left (near 0) and the attack distribution to the right (near 1) with minimal overlap.

What we observe:
- The **benign distribution** peaks sharply near 0 — the model confidently assigns low attack probability to most benign flows.
- The **attack distribution** is bimodal — one large cluster near 0 (attacks the model completely misses) and a second cluster approaching 1 (attacks it detects with high confidence).
- The overlap region (~0.3–0.7) represents the hard cases where the model is uncertain.

The bimodal attack distribution reflects the diversity of attack types in CIC-IDS2017: some attacks (e.g., DDoS, port scans) are easy to detect from packet statistics, while others (e.g., Infiltration, Web Attacks) produce traffic indistinguishable from benign at the flow level with our selected features.

---

### Calibration Curve

![Calibration Curve](calibration%20curve.png)

**Interpretation:** The calibration curve (reliability diagram) plots the model's predicted probability on the x-axis against the actual fraction of positive cases in each probability bin on the y-axis. A perfectly calibrated model lies on the diagonal.

The GNB calibration curve shows:
- **Overconfidence at high predicted probabilities** — when the model predicts >0.7 probability of attack, the actual attack rate is lower than predicted.
- **Underconfidence at mid-range probabilities** — flows assigned ~0.3–0.5 attack probability contain more attacks than the model believes.
- **Good calibration near zero** — the model correctly assigns very low probabilities to flows that are mostly benign.

This miscalibration is a known property of Naive Bayes: because the independence assumption is violated, predicted probabilities tend to be pushed toward 0 or 1 more than they should be (probability compression). The ROC AUC (0.761) tells us the ranking is reasonable even if absolute probabilities are off.

---

## Bayesian Logistic Regression

While GNB assumes feature independence, **Bayesian Logistic Regression (BLR)** is a discriminative model that learns a direct mapping from features to attack probability without the independence assumption.

$$P(y = 1 \mid \mathbf{x}, \boldsymbol{\beta}) = \sigma(\beta_0 + \boldsymbol{\beta}^\top \mathbf{x})$$

We place **Normal(0, 1) priors** on all coefficients, which encodes a mild regularizing belief that weights should be near zero absent data evidence:

$$\beta_0 \sim \mathcal{N}(0, 1), \quad \beta_j \sim \mathcal{N}(0, 1) \quad \forall j$$

### Automatic Differentiation Variational Inference (ADVI)

Rather than MCMC (which would take hours on 2.8M rows), we use **mean-field ADVI** (`pm.fit` in PyMC), which approximates the full posterior with a factored Gaussian:

$$q(\boldsymbol{\theta}) = \prod_j \mathcal{N}(\theta_j \mid \mu_j, \sigma_j^2)$$

ADVI optimizes the Evidence Lower Bound (ELBO) via gradient descent until convergence. The model is fit on a 5,000-row stratified subsample (4,041 benign, 959 attack) for tractability, and then 1,000 posterior samples are drawn from the approximation.

### ADVI Convergence

![ADVI Convergence](advi_convergence.png)

**Interpretation:** The convergence plot shows the negative ELBO over 30,000 ADVI iterations. A successful run shows:
- **Rapid initial descent** — the optimizer quickly finds a good approximate posterior in the first few thousand iterations.
- **Flattening plateau** — the ELBO stabilizes, confirming that optimization has converged and the mean-field approximation has settled.
- **Final negative ELBO ≈ 1,836.6** — this is a unitless measure of the quality of the variational approximation; lower (less negative) is better.

The smooth convergence without oscillation or divergence indicates that the Normal(0,1) priors are well-specified relative to the scaled features, and that ADVI is an appropriate inference method for this problem.

---

### Posterior Samples (Trace Plot)

![Posterior Samples](posterior_samples.png)

**Interpretation:** The trace plot shows the sampled values of each coefficient across the 1,000 posterior draws (left column) and their marginal posterior distributions (right column). Key observations:

- **Well-mixed traces** — the samples show no obvious trends, drifts, or stuck regions, indicating that the ADVI approximation has good support coverage.
- **Tight posterior distributions** — all coefficient posteriors are narrow Gaussians, meaning the data strongly informs the weights (high posterior precision).
- **Distinct coefficient magnitudes** — coefficients for `Packet Length Std` and `Max Packet Length` are large in magnitude, confirming their dominant influence on predictions.

---

### Posterior Coefficients (Forest Plot)

![Posterior Coefficients](posterior_coefficients.png)

**Interpretation:** The forest plot shows the posterior mean and 94% Highest Density Interval (HDI) for each regression coefficient. Coefficients whose HDI excludes zero are reliably nonzero given the data.

**MAP (Maximum A Posteriori) estimates for each coefficient:**

| Feature | MAP Coefficient | Interpretation |
|---|---|---|
| Intercept | −1.708 | Baseline log-odds strongly favor benign — consistent with the 80/20 class split |
| `Packet Length Std` | **+1.930** | Strongest positive predictor: high packet size variability strongly indicates attack |
| `Max Packet Length` | −1.299 | Negative: larger max packets alone are not attack-indicative after controlling for other features |
| `Fwd IAT Std` | +1.153 | High irregularity in forward inter-arrival times (automated traffic) signals attack |
| `Idle Mean` | +0.795 | Longer idle periods slightly increase attack probability |
| `Flow Duration` | −1.527 | Shorter flows (after standardization, negative coefficient = longer flow = less likely attack) |
| `Total Fwd Packets` | −1.404 | More forward packets slightly decreases attack probability, possibly due to DDoS single-packet floods |

**Key finding:** `Packet Length Std` is the dominant feature in both models — high within-flow packet length variability is the single most reliable indicator of malicious traffic in this dataset. `Fwd IAT Std` is the second strongest signal, capturing the irregular timing of automated attack tools.

The negative coefficients on `Max Packet Length` and `Flow Duration` are initially counterintuitive but make sense after standardization and controlling for other features: once `Packet Length Std` is accounted for, individual large packets or long flows are not independently predictive of attacks.

---

## Results Summary

### Gaussian Naive Bayes (from scratch)

| Metric | Value |
|---|---|
| Accuracy | 80.0% |
| ROC AUC | **0.761** |
| Precision (Attack) | 0.49 |
| Recall (Attack) | 0.45 |
| F1 Score (Attack) | 0.47 |
| Precision (Benign) | 0.87 |
| Recall (Benign) | 0.88 |
| F1 Score (Benign) | 0.88 |

### Bayesian Logistic Regression (PyMC + ADVI)

| Metric | Value |
|---|---|
| Inference method | ADVI (mean-field variational) |
| Iterations to convergence | 30,000 |
| Final negative ELBO | ~1,836.6 |
| MAP Intercept | −1.708 |
| Strongest coefficient | Packet Length Std (+1.930) |
| Posterior samples drawn | 1,000 |

### Model Comparison

| Model | Inference | Library deps | Uncertainty | Independence assumption |
|---|---|---|---|---|
| Gaussian Naive Bayes | Closed-form (exact) | NumPy only | No | Yes (violated here) |
| Bayesian Logistic Regression | ADVI (variational, ~seconds) | PyMC, Aesara | Full posterior | No |

---

## Conclusions

1. **Preprocessing is critical.** The CIC-IDS2017 dataset contains infinite values in flow rate features and leading/trailing whitespace in column names. Both issues cause silent errors if unaddressed.

2. **Class imbalance must be respected.** The 80/20 split means accuracy is a misleading metric. The Gaussian Naive Bayes achieves 80% accuracy almost by chance — recall on the attack class (45%) reveals the model misses the majority of intrusions.

3. **Packet statistics are the key signal.** Backward packet length variability (`Bwd Packet Length Std`, `Packet Length Std`) and temporal irregularity (`Fwd IAT Std`) are the features most discriminative of attack traffic. This is consistent with the literature: attack tools generate statistically distinct traffic patterns compared to human-driven sessions.

4. **The Bayesian advantage.** Both models return calibrated probability scores rather than hard labels, enabling threshold tuning based on operational requirements (e.g., a security team may tolerate more false positives to catch more attacks). The BLR additionally quantifies uncertainty over each feature's contribution — the HDI plots show which relationships are robust and which are uncertain.

5. **ADVI vs. MCMC.** ADVI approximates the posterior via gradient-based optimization of the ELBO in seconds, vs. hours for NUTS/MCMC on this dataset. The mean-field approximation slightly underestimates posterior variance (it assumes independent coefficients), but the point estimates align well with MAP, confirming convergence.

6. **GNB limitations.** The conditional independence assumption is clearly violated — packet length features are strongly intercorrelated (r > 0.9 in several pairs). This leads to the bimodal posterior probability distribution and poor calibration observed above. Despite this, GNB achieves ROC AUC = 0.761, showing that even violated assumptions can produce useful probabilistic rankings.

7. **Future work.** Directions for improvement include: (a) feature engineering on temporal patterns (flow sequences); (b) class-weighted or oversampled training to improve attack recall; (c) full MCMC inference using NUTS on a larger subsample; (d) hierarchical Bayesian models grouping flows by source IP or attack day.

---

## References

- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*. ICISSP.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
- Gelman, A., Carlin, J., Stern, H., Dunson, D., Vehtari, A., & Rubin, D. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
- Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M. (2017). *Automatic Differentiation Variational Inference*. JMLR.
- PyMC Development Team. [pymc.io](https://www.pymc.io)
- ArviZ Development Team. [python.arviz.org](https://python.arviz.org)
