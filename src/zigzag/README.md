## ZigZag persistence algorithm 


To run this code with the FastZigZag algorithm ([paper](https://arxiv.org/abs/2204.11080)) install the dependencies for it by following the instruction on their [GitHub repository](https://github.com/TDA-Jyamiti/fzz).

Note that since the installation is not out of the box, it is not provided automatically by the conda environment of this project.

### run the ZigZag algorithm
```bash
python run_fast_zigzag.py --reps_path="<path/to/extracted/representations>" \
                          --knn=5 \
                          --dim=4 \
                          --output_folder="<output/path/to_folder>" \
                          --output_file="output_file.csv"


```