---
title: "bayes_oracle: a test case where the right answer is known"
---

# What this folder shows

With real traces, nobody knows the best possible success rate, so there is no way to tell how close a certificate is to the truth. This folder builds artificial traces where the best possible success rate is known exactly, and compares the BI of a set of models with that true value.

- `gaussian_oracle_suite.py`: each of the eight bits of the key byte leaks through its own noisy channel. The best possible attacker and its success rate are known in closed form. The paper reports a true rate of 0.989577 and a measured suite rate of 0.989503. Level A runs this script and checks both numbers.
- `bayes_oracle_synthetic.py`: a second artificial case with masking, used to compare networks of different widths.

# Before you start

Only `requirements.txt`. No data. About twenty seconds.

# Run

See the options:

```bash
python -m bayes_oracle.gaussian_oracle_suite --help
```

```{.text transcript="transcripts/02_bayes_oracle/help.txt"}
```

Run with the paper's settings (they are also the defaults):

```bash
python -m bayes_oracle.gaussian_oracle_suite --amplitude 3.0 --sigma 1.0 \
    --n-train 102400 --n-attack 100000 --n-seeds 3 --output-dir results/bayes_gaussian_oracle_suite
```

```{.text transcript="transcripts/02_bayes_oracle/run.txt"}
```

```{.text transcript="transcripts/02_bayes_oracle/files.txt"}
```

```{.text transcript="transcripts/02_bayes_oracle/summary.txt"}
```

# What to look for

The two numbers to compare are in the summary table. The true success rate is 0.989577 and the suite's measured rate is 0.989503. The gap is 0.00007, which shows that the suite is as good as the best possible attacker, up to the small margin that comes from testing on a finite number of traces.
