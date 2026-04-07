# Measuring Tool Precision Faithfulness in ReAct Agents

**Author:** Manoj  
**Status:** Preprint — submitted to arXiv  
**Target:** NeurIPS 2026 LLM Agents Workshop  

## Overview

This repository contains the code and data for the paper:

> "Measuring Tool Precision Faithfulness in ReAct Agents"

We introduce **Tool Precision Faithfulness** — a metric for measuring whether a ReAct agent's stated reasoning faithfully predicts its actual tool selection, guided by the **Minimum Sufficiency Principle**.

## Key Findings

| Category | Tasks | Faithfulness | Min Sufficiency |
|---|---|---|---|
| Simple arithmetic | 30 | 100% | 100% |
| Complex functions | 30 | 100% | 100% |
| Ambiguous operations | 30 | 100% | 90% |
| Delegation | 10 | 100% | 100% |
| **Overall** | **100** | **100%** | **97%** |

## Novel Contributions

1. **Tool Precision Faithfulness** metric
2. **Minimum Sufficiency Principle** for tool selection
3. **Power Operation Bias** — agents associate mathematical power with computational power
4. **Sqrt Operation Bias** — agents over-route perfect squares to powerful tools
5. **Delegation Resistance** — agents correctly ignore "use most powerful tool" instructions

## Repository Structure

```
react-faithfulness/
├── README.md
├── react_faithfulness_experiment.py   # Main experiment code
├── requirements.txt                    # Dependencies
└── results/
    └── faithfulness_results.json      # Experiment results
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
python react_faithfulness_experiment.py
```

## Requirements

```
langgraph
langchain
langchain-core
anthropic
```

## Citation

```bibtex
@article{manoj2026toolprecision,
  title={Measuring Tool Precision Faithfulness in ReAct Agents},
  author={Manoj},
  journal={arXiv preprint},
  year={2026}
}
```

## AI Assistance Disclosure

This research was developed with AI assistance (Claude, Anthropic) for writing, coding, and experimentation. All research ideas, insights, analogies, and experimental design originated with the human author.
