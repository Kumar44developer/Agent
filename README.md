<div align="center">

# Delta Twitter Support Agent

**A trustworthy AI customer-support agent that classifies intent, drafts a grounded reply, and decides when to escalate to a human — with evidence, not vibes.**

Built for a single brand (Delta Air Lines) on the Kaggle *Customer Support on Twitter* corpus,
with a rigorous evaluation harness, a hand-labelled golden set, and an LLM-as-judge whose agreement
with humans is measured rather than assumed.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)
![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen?style=flat-square)
![Tests](https://img.shields.io/badge/tests-26%20passing-success?style=flat-square)
![Cross-platform](https://img.shields.io/badge/OS-Windows%20·%20macOS%20·%20Linux-lightgrey?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)

</div>

---

## Table of Contents

- [Why This Project](#why-this-project)
- [What The Agent Does](#what-the-agent-does)
- [Highlights](#highlights)
- [Quick Start](#quick-start)
- [Try It On a Message](#try-it-on-a-message)
- [Headline Results](#headline-results)
- [How It Works](#how-it-works)
- [Running On The Real Kaggle Data](#running-on-the-real-kaggle-data)
- [Optional Hosted LLM Backend](#optional-hosted-llm-backend)
- [Project Layout](#project-layout)
- [Data](#data)
- [Configuration](#configuration)
- [Tests](#tests)
- [Reading The Numbers Honestly](#reading-the-numbers-honestly)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Author](#author)
- [License](#license)

---

## Why This Project

Most "AI agent" demos stop at a prompt and a hope. This one is built around a single question:
**can you show, with evidence, that the agent is reliable enough to act on?** That focus drives
everything downstream — an evaluation harness, a hand-labelled golden set split into dev and a
held-out test, honest baselines, a judge validated against human labels with a measured Cohen's κ,
and a candid failure analysis. The agent is also deliberately **zero-dependency** so its headline
results reproduce on any machine with a stock Python install.

---

## What The Agent Does

Given an inbound customer tweet, the agent:

1. **Classifies intent** into a compact, action-oriented taxonomy of 9 intents.
2. **Drafts a reply** grounded in how the brand has historically resolved similar issues.
3. **Decides auto-handle vs. escalate** to a human, with a stated, auditable reason.

---

## Highlights

- **Deterministic and dependency-free** — TF-IDF, Naive Bayes, a logistic-regression student, retrieval, all metrics, and Cohen's κ are implemented against the Python standard library only
- **Trust-first evaluation** — a 200-example golden set, a stratified dev/test split, and a held-out test number scored only once
- **Measured judge** — LLM-as-judge agreement with humans validated on 36 hand-labelled verdicts (κ = 0.94), not taken on faith
- **Honest baselines** — trivial, keyword-rule, rule-chain, learned-student, and the shipped ensemble are all reported side by side
- **Plug-in LLM backends** — a deterministic default with no API key, plus OpenAI and Anthropic adapters that activate when a key is present
- **Cross-platform** — runs identically on Windows, macOS, and Linux
- **100% test coverage of the core** — 26 unit and end-to-end tests, all green

---

## Quick Start

Requirements: **Python 3.10 or newer** and nothing else. Reproduce the headline results in well
under 15 minutes.

```bash
python cli.py setup
python cli.py eval
```

The first `eval` trains the learned student and caches its weights under `.cache/`; later runs reuse
the cache and finish in seconds. The cache key includes the corpus size and every hyperparameter,
so any change retrains automatically.

`python cli.py eval` prints the headline tables and writes:

- `results/results.json` — every metric, machine-readable
- `results/summary.md` — the same results, human-readable
- `results/confusion_main.txt` — the intent confusion matrix

---

## Try It On a Message

```bash
python cli.py demo
python cli.py handle "@Delta my flight got cancelled, stuck at JFK, need to rebook"
```

Example output:

```text
IN : @Delta my flight got cancelled, stuck at JFK, need to rebook
INTENT: flight_disruption  (conf=0.99)   ROUTE: ESCALATE
WHY : Escalate to human — time_critical_or_vulnerable; high_stakes_intent:flight_disruption.
REPLY: So sorry for the disruption to your travel plans — that's stressful. We'll check the next
       available flights. Please DM us your confirmation number so we can help.
GROUNDED: True  | top evidence (sim=0.311): "We're so sorry for the disruption. Please DM us..."
```

---

## Headline Results

Computed on the bundled sample corpus with the default zero-dependency backend and judge.

**Intent classification.** The 200-example golden set is split into a **dev half used for all
tuning** and a **held-out test half scored only once, at the end**. The held-out number is the one
to trust.

| split | n | trivial | keyword rules | rule chain | learned student | shipped ensemble |
|---|---|---|---|---|---|---|
| dev | 119 | 16.0% | 55.5% | 93.3% | 86.6% | 89.1% |
| test (held out) | 81 | 14.8% | 55.6% | 79.0% | 74.1% | **81.5%** |
| test ∩ never-inspected | 17 | 17.6% | 70.6% | 88.2% | 76.5% | 88.2% |
| whole set | 200 | 15.5% | 55.5% | 87.5% | 81.5% | **86.0%** |

The main model is an **equal-weight ensemble** of a hand-written rule chain and a learned model
distilled from it. Read the table this way: the rule chain looks best on dev *because it was
hand-tuned there*, while the student never saw dev. On held-out test the ensemble beats both, and
its dev→test gap is **7.6 points versus 13.9** for the rule chain alone — ensembling recovered the
generalization that hand-tuning had cost.

**Escalation decision** (positive class = escalate, whole set):

| policy | precision | recall | F1 | auto-handle rate | missed-escalation rate |
|---|---|---|---|---|---|
| trivial (escalate everything) | 0.59 | 1.00 | 0.74 | 0% | 0% |
| simple (intent-only rule) | 0.80 | 0.41 | 0.54 | 60% | 34.5% |
| **full agent** | **0.83** | **0.68** | **0.75** | **49%** | **18.5%** |

**Reply quality** (scored by the judge): **93.0%** acceptable, **72.5%** grounded in retrieved
history, where a reply counts as grounded only when the retrieved exemplar shares the predicted
intent.

**Judge trustworthiness**: across 36 hand-labelled reply verdicts, judge↔human agreement is
**97.2%** with **Cohen's κ = 0.94**.

---

## How It Works

The pipeline runs end to end in pure Python:

1. **Data loading** reconstructs customer-agent threads from the raw tweet CSV.
2. **Signals and features** turn each inbound message into normalized tokens, character n-grams, and hand-crafted risk signals (PII, legal language, urgency, account access).
3. **Intent classification** blends a rule chain with a logistic-regression student distilled from it into a single ensemble prediction and confidence.
4. **Retrieval** finds the most similar historical resolutions with a TF-IDF cosine search.
5. **Reply generation** drafts a response grounded in the retrieved exemplar through a pluggable LLM backend.
6. **Escalation** weighs confidence, stakes, and risk signals into an auto-handle or escalate decision with a human-readable reason.
7. **Evaluation** scores all three capabilities on the golden set and validates the judge against human labels.

---

## Running On The Real Kaggle Data

Download `twcs.csv` from Kaggle, point the pipeline at it, and run the same evaluation. Nothing
else changes — the same thread builder, classifier, retriever, escalation policy, and harness run
against real data.

```bash
export TWCS_PATH=/path/to/twcs.csv
python cli.py eval
```

---

## Optional Hosted LLM Backend

Replies and judge verdicts default to a deterministic, no-key backend so results are stable
run-to-run. To generate and judge with a hosted model instead, set the backend and key; the API
calls use the standard-library `urllib`, so there is still nothing to install.

```bash
export SUPPORT_AGENT_LLM=openai
export OPENAI_API_KEY=sk-your-key-here
export SUPPORT_AGENT_LLM_MODEL=gpt-4o-mini
python cli.py demo
```

---

## Project Layout

```text
Agent/
├── cli.py
├── src/support_agent/
│   ├── agent.py
│   ├── config.py
│   ├── intents.py
│   ├── data_loader.py
│   ├── text.py
│   ├── vectorizer.py
│   ├── signals.py
│   ├── features.py
│   ├── retriever.py
│   ├── reply.py
│   ├── escalation.py
│   ├── classifiers/
│   │   ├── trivial.py
│   │   ├── rules.py
│   │   ├── nb.py
│   │   ├── hybrid.py
│   │   ├── rules_refined.py
│   │   ├── refined.py
│   │   ├── logreg.py
│   │   └── stacked.py
│   └── llm/
│       └── backend.py
├── eval/
│   ├── metrics.py
│   ├── splits.py
│   ├── judge.py
│   └── run_eval.py
├── scripts/
│   ├── generate_sample_data.py
│   ├── build_golden.py
│   └── build_judge_set.py
├── data/
├── tests/
├── results/
├── report/REPORT.md
└── DECISIONS.md
```

---

## Data

| File | What it is |
|---|---|
| `data/raw/twcs_sample_delta.csv` | Self-contained sample corpus in the Kaggle `twcs.csv` schema, produced by `scripts/generate_sample_data.py`, so the pipeline runs end-to-end without the full download |
| `data/golden/golden_eval.jsonl` | 200 hand-labelled examples (intent plus escalation), split into dev and test by `eval/splits.py` |
| `data/golden/judge_agreement.jsonl` | 36 hand-labelled reply-quality verdicts used to validate the judge |
| `data/golden/README.md` | How the golden sets were sampled and labelled |

---

## Configuration

Runtime behaviour is driven by a small set of environment variables, all optional:

| Variable | Default | Effect |
|---|---|---|
| `TWCS_PATH` | bundled sample CSV | Path to the real Kaggle `twcs.csv` |
| `SUPPORT_AGENT_LLM` | deterministic | Set to `openai` or `anthropic` to use a hosted model |
| `SUPPORT_AGENT_LLM_MODEL` | backend default | Model name for the hosted backend |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | unset | Credentials; the matching backend activates when present |

---

## Tests

The full suite runs against the standard library with no installation step.

```bash
python -m unittest discover -s tests -v
```

26 tests cover text normalization, vectorizer cosine math, the feature pipeline, deterministic
logistic regression, the escalation policy, the dev/test split, and an end-to-end agent pass that
asserts valid intents and that replies never solicit card or password details.

---

## Reading The Numbers Honestly

The headline metrics are computed on the bundled sample, which is cleaner and less diverse than the
full Twitter corpus. Treat them as an **upper bound** until re-measured on the real data. The
project's own report calls this out first, alongside the "what is misleading about my headline
number" analysis — this candor is a feature, not a caveat.

---

## Documentation

- **`report/REPORT.md`** — the full write-up: problem framing, results, failure modes, and next steps
- **`DECISIONS.md`** — the decision log behind the design trade-offs

---

## Contributing

Contributions are welcome.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-change`)
3. Add or update tests so the suite stays green
4. Commit your changes (`git commit -m "Add amazing change"`)
5. Push to the branch (`git push origin feature/amazing-change`)
6. Open a Pull Request

---

## Author

Created by **[Kumar44developer](https://github.com/Kumar44developer)**.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
