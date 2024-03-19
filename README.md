# ChromBERT: Uncovering Chromatin State Motifs in the Human Genome using a BERT-based Approach

This repository contains the code for 'ChromBERT: Uncovering Chromatin State Motifs in the Human Genome using a BERT-based Approach'. 
If you utilize our models or code, please reference our paper. We are continuously developing this repo, and welcome any issue reports.

This package offers the source codes for the ChromBERT model, which draws significant inspiration from [DNABERT](https://doi.org/10.1093/bioinformatics/btab083) 
(Y. Ji et al., "DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome", Bioinformatics, 2021.) 
It includes pre-trained models for promoter regions, 
fine-tuned models, and a motif visualization tool. Following the approach of DNABERT, training encompasses general-purpose pre-training 
and task-specific fine-tuning. We supply pre-trained models for both 2k upstream and 4k downstream, as well as 4k upstream and 4k downstream configurations. 
Additionally, we provide a fine-tuning example for the 2k upstream and 4k downstream setup. For data preprocessing, utility functions are available in `utility.py`, 
which users can customize to meet their requirements.

## Citation

## Environment Setup

For the environment setup, including the Python version and other settings, you can refer to the configurations used in the [DNABERT GitHub repository](https://github.com/jerryji1993/DNABERT). These guidelines will help ensure compatibility with the foundational aspects we utilize from DNABERT.

