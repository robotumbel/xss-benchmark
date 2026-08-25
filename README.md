# XSS detection under extreme class imbalance — replication package

Code, configurations and results for

> **Cross-Site Scripting Detection under Extreme Class Imbalance: A Leakage-Controlled Cross-Representation Benchmark**
> B. Purnama, E. A. Winanto, Sharipuddin. *Informatics and Automation (SPIIRAS Proceedings)*, 2026.

The paper compares ten models on 67 engineered structural features under a
100:1 benign-to-attack ratio and seven models on 42,671 raw payloads, under a
protocol in which preprocessing statistics and decision thresholds are derived
only from training and validation data. Every table and figure in the paper is
produced by a script in `code/`.

## What is here

```
code/       experiment scripts (see the table below)
results/    the CSV output of every script, exactly as used in the paper
figures/    the seven manuscript figures
```

Datasets are **not** redistributed here; all three are public (see *Data*).

## Protocol

`code/rev_common.py` implements the rules that every experiment follows:

- stratified 70/10/20 train/validation/test split preserving the class ratio;
- feature standardisation fit on the training split only;
- decision threshold chosen to maximise F1 on validation, then applied
  unchanged to the test split;
- ranking by AUPRC and MCC, with bootstrap 95% confidence intervals;
- McNemar tests for pairwise comparison, with Holm–Bonferroni correction over
  the family of comparisons.

## Experiments

| Script | What it measures |
|--------|------------------|
| `rev_r1_duplicates.py` | duplicate and cross-split feature-vector audit; re-run of the ranking on a de-duplicated corpus |
| `rev_r2_imbalance_sweep.py` | the full protocol at benign:attack ratios of 100:1, 250:1, 500:1, 1000:1, 2000:1 and 5000:1, three seeds |
| `rev_r3r4_capacity_imbalance.py` | trainable-parameter accounting; recurrent family matched by parameter count instead of hidden size; 2×2 ablation of class-weighted loss against minority oversampling |
| `rev_r5_stats_threshold.py` | all-pairs McNemar with Holm correction; the operating point under a validation-calibrated threshold against test-set F1 and F2 oracles |
| `rev_r7_modern_dl.py` | Char-CNN, Char-BiLSTM, attention BiLSTM and a character Transformer on raw payloads, three seeds, under clean, obfuscated and artifact-suppressing conditions |
| `rev_r7b_crossdataset_modern.py` | cross-corpus transfer for the attention and transformer models |
| `rev_r8_obfuscation.py` | three evasion tiers with a benign control and a test of how detectable the transformation itself is |
| `rev_r9_alt_feature_schema.py` | replication of the model ranking on an independently designed 32-feature schema built from a different corpus |
| `rev_r10_shap_rules.py` | SHAP attribution in log-odds units, attack-marker against document-context ablation, and a distilled decision list |
| `rev_r11_tuning_cost.py` | development cost of a tree ensemble against a recurrent model under an identical search budget |
| `rev_figs.py` | regenerates the imbalance-sweep and evasion figures |

The scripts prefixed `xss_` are the original pipeline from the first version of
the study (Study A ranking, raw-payload models, obfuscation, cross-dataset
transfer, calibration, and the grayscale figure generation).

## Data

| Corpus | Used for | Source |
|--------|----------|--------|
| Cross-Site Scripting (XSS) Attack Dataset — 101,000 samples, 67 structural features, 100:1 | Study A | released with Mokbal et al., *XGBXSS*, JISA 58:102813, 2021 |
| Mereani & Howe payload corpus — 42,671 payloads | Study B, source A | Mereani & Howe, AISC 845, Springer, 2019 |
| HTML-form payload corpus — 13,686 payloads after de-duplication | Study B, source B | public Kaggle release (`syedsaqlainhussain/cross-site-scripting-xss-dataset-for-deep-learning`) |

Place `XSS_dataset1.csv` and a `data_raw/` directory (holding
`xss_payloads_clean.parquet` and `xss_syedsaqlain.csv`) in one directory, then
point the scripts at it:

```bash
export XSS_DATA_DIR=/path/to/datasets
```

Without that variable the scripts look two directories above `code/`, which is
the authors' working layout.

## Running

```bash
pip install -r requirements.txt
cd code
python rev_r1_duplicates.py
```

Every script writes its CSV output to `../rev_out/` (create it, or copy the
provided `results/` there) and prints the same numbers to standard output.
Approximate runtimes on one NVIDIA RTX 4050 laptop GPU with a 12-core CPU:
R1 two minutes, R2 forty minutes, R3/R4 twenty-five minutes, R5 seconds,
R7 fifteen minutes, R7b ten minutes, R8 three minutes, R9 five minutes,
R10 three minutes, R11 ten minutes.

Results are seed-dependent in one respect that the paper reports explicitly:
recall under artifact-suppressing evasion varies by up to 0.68 across training
seeds of the same model, while its in-distribution F1 varies by 0.0003. All
out-of-distribution numbers are therefore reported as means over three seeds
with their standard deviation, and single-run numbers should not be compared.

## Citation

```bibtex
@article{purnama2026xss,
  title   = {Cross-Site Scripting Detection under Extreme Class Imbalance:
             A Leakage-Controlled Cross-Representation Benchmark},
  author  = {Purnama, Benni and Winanto, Eko Arip and Sharipuddin},
  journal = {Informatics and Automation},
  year    = {2026}
}
```

## License

MIT, see `LICENSE`. The datasets remain under the terms set by their original
publishers.
