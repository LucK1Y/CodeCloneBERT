# Code Clone BERT for Code Clone Detection: A Replication of a Replication Study
This repository contains the code for the Seminar **Empirical Software Engineering** at [University of Zurich](https://www.uzh.ch/en.html).

I present a method for fine-tuning a RoBERTa model for code clone detection, a binary classification task that determines whether two pieces of code have the same functionality, even if they have different syntax. The fine-tuned model is available at [hugginface](https://huggingface.co/4luc/codebert-code-clone-detector).


## Table of Contents
* [Introduction](#introduction)
* [Overview directories](#overview-directories)
* [Dataset](#dataset)
* [Results](#results)
* [References](#references)
* [Appendix](#appendix)
<!-- * [Contact](#contact) -->

## Introduction
Code clones, also known as code duplication, are identical or similar code fragments that appear in multiple places within a software system. These clones are not a new phenomenon, but rather a natural byproduct of software development. They can originate from various sources, including:

* Copy-paste programming: a common practice among developers to reuse existing code, often with minor modifications.
* Independent development: multiple developers working on similar functionality, unaware of each other's efforts.
* Legacy code: inherited codebases that have evolved over time, with duplicated code remaining unnoticed.

Code clones are a problem because they can lead to:

* Maintenance nightmares: changes to one clone may not be reflected in others, introducing inconsistencies and errors.
* Increased code size: duplicated code consumes more memory and disk space, affecting system performance.
* Higher bug density: clones can harbor identical bugs, making debugging more challenging.

Detecting code clones is essential to ensure the quality, reliability, and maintainability of software systems. By identifying and refactoring clones, developers can reduce technical debt, improve code readability, and make their systems more efficient.

#### Why Deep learning and CodeBERT
**Leveraging AI to Uncover Hidden Duplicates**

Traditional code clone detection methods rely on simple string matching, syntax analysis, or token-based comparisons. However, these approaches have limitations, such as:

* High false positive rates: incorrectly identifying similar code as clones.
* Low detection rates: missing subtle or obfuscated clones.
* Limited scalability: struggling to handle large codebases or complex syntax.

To overcome these challenges, researchers have turned to deep learning techniques, which have revolutionized various fields, including natural language processing (NLP) and computer vision. The application of deep learning to code analysis has given rise to powerful models like CodeBERT, specifically designed for code understanding and clone detection.

**CodeBERT: A Game-Changer in Code Analysis**

CodeBERT is a pre-trained language model that leverages the transformer architecture, popularized by BERT (Bidirectional Encoder Representations from Transformers) and RoBERTa. By fine-tuning CodeBERT on large code datasets, it learns to represent code snippets as dense vectors, capturing their semantic meaning and relationships.

CodeBERT's advantages for code clone detection include:

* **Contextual understanding**: CodeBERT considers the surrounding code context, reducing false positives and improving detection accuracy.
* **Robustness to syntax variations**: CodeBERT is trained on diverse code styles and syntax, making it more resilient to obfuscation and minor modifications.
* **Scalability**: CodeBERT can handle large codebases and complex code structures, making it an ideal solution for industrial-scale software systems.

By employing CodeBERT for code clone detection, developers can:

* Identify clones with higher accuracy and precision
* Reduce manual effort and false positives
* Improve code quality and maintainability
* Enhance the overall development experience

## Organization
In this repository, I have primarily stored Jupyter notebooks for all stages of the experiment like preparing the datasets and fine-tuning the model. I used kaggle for the fine-tuning.

#### Overview directories
````
.
├── 0_dataset_creations                                         # Use these scripts to recreate the datasets
├── 1_re_run_BigCloneBench_model_old                            # fist replication
├── 2_fine-tune_semantic                                        # second replication
└── 3_fine-tune_gptCloneBench                                   # third novel replication with a new dataset

4 directories
````

#### Tokenization
The file **custome_tokenizers.py** contains a diverse set of tokenization methods. As I tried to get the same tokenization used in (Saad, 2022), I found the following code to be best effective.
- "miss"- use the batch functionality here, to tokenize both clone pairs in a batch of two.
- set **max_length** to 255. Theoretical it supports 257 tokens but changes from BERT to RoBERTa limit the window to 255 tokens. ("Improvement" in the positional encoding.)
````python
def tokenization(row):
    tokenized_inputs = tokenizer(
        [row["clone1"], row["clone2"]],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=255,
    )
    tokenized_inputs["input_ids"] = tokenized_inputs["input_ids"].flatten()
    tokenized_inputs["attention_mask"] = tokenized_inputs["attention_mask"].flatten()
    return tokenized_inputs
````

# Dataset
Please note that the datasets used in this project, Semanticclonebench and Gptclonebench, have specific licensing restrictions. The Semanticclonebench dataset is based on Stackoverflow answers, which may violate software licenses in some cases. Meanwhile, the Gptclonebench dataset is licensed by OpenAI for non-commercial use only.

To respect these licensing constraints, I do not publicly release the resulting datasets. However, I provide the necessary scripts in this repository, allowing you to recreate the datasets on your own.

# Results
Here is a table showing some results on the test split of SCB:

| **Experiment** | **Precision** | **Recall** | **F1-Score** | **Accuracy** | **Training Steps** |
| --- | --- | --- | --- | --- | --- |
| CodeBERT<sub>SCB - tuned</sub> | 0.888 | 0.942 | 0.914 | 0.913 | 536 |
| CodeBERT<sub>GPTCB - tuned</sub> | **0.977** | 0.863 | 0.917 | 0.923 | 2720 |
| CodeBERT<sub>GPTCB - tuned; SCB - tuned</sub> | 0.919 | **0.967** | **0.942** | **0.941** | 2052; 396 |

# References
- Farouq Al-Omari, Chanchal K. Roy, and Tonghao Chen. **Semanticclonebench**: A semantic
code clone benchmark using crowd-source knowledge. In 2020 IEEE 14th International
Workshop on Software Clones (IWSC), pages 57–63, 2020.
    - Dataset: https://drive.google.com/open?id=1KicfslV02p6GDPPBjZHNlmiXk-9IoGWl
- Ajmain I. Alam, Palash R. Roy, Farouq Al-Omari, Chanchal K. Roy, Banani Roy, and
Kevin A. Schneider. **Gptclonebench**: A comprehensive benchmark of semantic clones and
cross-language clones using gpt-3 model and semanticclonebench. In 2023 IEEE Interna-
tional Conference on Software Maintenance and Evolution (ICSME), pages 1–13, 2023.
    - Dataset: https://shorturl.at/jvxOV
- Saad Arshad, Shamsa Abid, and Shafay Shamail. Codebert for code clone detection: A
replication study. In 2022 IEEE 16th International Workshop on Software Clones (IWSC),
pages 39–45, 2022.
    - CodeBERT for Code Clone Detection Replication Pack. https://doi.org/10.5281/zenodo.6361315, 2022.
- Feng, Z., Guo, D., Tang, D., Duan, N., Feng, X., Gong, M., Shou, L., Qin, B., Liu, T., Jiang, D., & Zhou, M. (2020). CodeBERT: A Pre-Trained Model for Programming and Natural Languages. arXiv preprint arXiv:2002.08155, 2020.
    - Model Repository: https://github.com/microsoft/CodeBERT

# Appendix
#### Disclaimer on Generative AI
I rephrased some sentences using Generative AI as well as for Coding Support.