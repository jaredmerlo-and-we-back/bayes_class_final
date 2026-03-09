# Bayesian Intrusion Detection with CIC-IDS2017

**Course:** Bayesian Machine Learning
**Dataset:** [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) — Canadian Institute for Cybersecurity

---

## Abstract

Modern intrusion detection systems (IDS) rely heavily on rule-based signatures that fail to generalize to novel attack patterns. This project applies **Bayesian probabilistic modeling** to the CIC-IDS2017 network traffic dataset to classify flows as benign or malicious.

We implement and compare two models:

1. **Gaussian Naive Bayes (from scratch)** — a generative Bayesian classifier built without any ML libraries, using Bayes' theorem with Gaussian class-conditional likelihoods.
2. **Bayesian Logistic Regression (PyMC + MCMC)** — a discriminative model with Normal(0,1) priors on all coefficients, fit via the NUTS sampler, yielding full posterior distributions over model weights.

The dataset contains ~2.8 million labeled network flow records spanning one week of simulated traffic, including eight attack categories (DDoS, port scanning, web attacks, infiltration, and more).

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

> **Note:** The CSV files are not tracked in this repository because of their size (~1 GB total). Download from the [CIC website](https://www.unb.ca/cic/datasets/ids-2017.html) and place them in `MachineLearningCSV/MachineLearningCVE/`.

---

## Methods

### 1. Gaussian Naive Bayes (from scratch)

Applies Bayes' theorem under the conditional independence assumption:

$$P(c \mid \mathbf{x}) \propto P(c) \prod_{i=1}^{d} P(x_i \mid c)$$

Each class-conditional likelihood is modeled as a Gaussian:

$$P(x_i \mid c) = \frac{1}{\sqrt{2\pi\sigma_{i,c}^2}} \exp\!\left(-\frac{(x_i - \mu_{i,c})^2}{2\sigma_{i,c}^2}\right)$$

The log-sum-exp trick is used at inference for numerical stability. All parameters ($\mu$, $\sigma^2$, and class priors) are estimated directly from training data.

### 2. Bayesian Logistic Regression (PyMC)

A discriminative model placing Normal priors on all coefficients:

$$P(y=1 \mid \mathbf{x}) = \sigma(\beta_0 + \boldsymbol{\beta}^\top \mathbf{x}), \quad \beta_j \sim \mathcal{N}(0, 1)$$

Inference is performed via **ADVI (Automatic Differentiation Variational Inference)** using PyMC. ADVI approximates the posterior by optimizing the Evidence Lower Bound (ELBO) with gradient descent, converging in seconds vs. hours for MCMC. The result is a full approximate posterior distribution over each weight. MAP estimation is also computed for comparison.

---

## Features Used

| Feature | Rationale |
|---|---|
| `Packet Length Std` | High variance in packet sizes is characteristic of attack traffic |
| `Max Packet Length` | Attacks often involve unusually large packets |
| `Fwd IAT Std` | Irregular inter-arrival times signal automated (non-human) traffic |
| `Idle Mean` | Attack flows have distinct idle behavior compared to benign |
| `Flow Duration` | Short, high-volume flows are typical of DDoS and scan attacks |
| `Total Fwd Packets` | Volume of forward packets distinguishes attack types |

Features are standardized (zero mean, unit variance) before model fitting.

---

## Results

| Metric | Gaussian Naive Bayes |
|---|---|
| ROC AUC | see `main.ipynb` |
| Precision (Attack) | see `main.ipynb` |
| Recall (Attack) | see `main.ipynb` |
| F1 Score (Attack) | see `main.ipynb` |

> Full outputs, plots, and MCMC posterior summaries are in [`main.ipynb`](./main.ipynb).

---

## References

- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization*. ICISSP.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
- PyMC Development Team. [pymc.io](https://www.pymc.io)
"# bayes_class_final" 
