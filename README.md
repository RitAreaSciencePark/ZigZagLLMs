# Persistent Topological Features in Large Language Models
## Arxiv paper: *****


With this repo is it possibile to reproduce the data of the paper **Persistent topological features in large language models**
-   **representation**: folder for the extraction of the hidden representations.
-   **zigzag** :folder for the execution fo the ZigZag algorithm over the representations.
-   **benchmark**: folder where it is possible to run the benchmarks with the different prunig methods.
    -   Benchmark with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
    -   Extraction of the blocks to cut with the [Angular distance](https://arxiv.org/abs/2403.17887v1) and [Bi-Score](https://arxiv.org/abs/2403.03853) are done with a modified version of [short-transformers](https://github.com/melisa/short-transformers) (to make it run also with Pyhia 6.9B).
-   **plots**: folder where it is possible to reproduce the plots of the paper.
## Environment setup

To install the library for FastZigZag refer to their [paper](https://arxiv.org/abs/2204.11080) and their [github folder](https://github.com/TDA-Jyamiti/fzz).

To install the rest of the environment ```conda env create -f environment.yml```

