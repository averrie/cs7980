# Bias Detection in Political Social Media

This repository contains code and notebooks for the project **Bias Detection in Political Social Media**, a cross-platform study of political bias in short text using data from **Truth Social** and **Bluesky**, with additional experiments on **U.S. Congressional tweets (2008–2017)** and the **Politics.com (2005)** dataset.

The project builds an LLM-annotated corpus (~130k posts) with Left / Neutral / Right labels, benchmarks Naive Bayes and DistilBERT classifiers, and analyzes cross-platform semantics and temporal / domain shift.

> **Note:** Raw datasets are not included in this repository. The main labeled corpus is available on Hugging Face: [averrie/bluesky-truthsocial-political-stances](https://huggingface.co/datasets/averrie/bluesky-truthsocial-political-stances)

## Citing

If you use this code or dataset in academic work, please cite:
```bibtex
@misc{huang2025bias-political-social-media,
  title        = {Bias Detection in Political Social Media},
  author       = {Huang, Xingrui and Xu, Liangyou},
  year         = {2025},
  howpublished = {\url{https://github.com/averrie/cs7980}},
}
```
