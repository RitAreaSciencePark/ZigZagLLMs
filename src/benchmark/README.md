## Benchmarks


In this folder it is possible to run benchmarks with the suite provided by EleutherAI
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

Remember that in this work we prune by cutting blocks of layers.

For a given block it is possible to get the starting point with different methods like the [Angular distance](https://arxiv.org/abs/2403.17887v1) and [Bi-Score](https://arxiv.org/abs/2403.03853). 
We provide a temporary modified version of [short-transformers](https://github.com/melisa/short-transformers) to run Pythia 6.9B.


### Run benchmark

The inputs for the ```--task``` parameter are the benchmarks of **lm-evaluation-harness** in our case: mmlu, hellaswag or winogrande.

It is possible to run the benchmarks in three possible setup by deciding which algorithm to use for pruning. 


-   ```--persistence``` : our algorithm which will perform the benchmark with a 10% cut and a 20% cut, refer to the paper for more details.
-   ```--full```        : benchmark on the full model without pruning.
-   ```--personalized```: the benchmark will be made by cutting a block that will start and end from the layers given by the user.
    - ```--start_block```: starting layer from which the cut block begins.
    - ```--end_block```: last layer that will be cut.


### Our method
```bash
python benchmark.py --persistence \
                    --model="meta-llama/Llama-2-7b-hf"\
                    --task="mmlu"\
                    --zigzag_output="<path/to/the/zigzag/csv-file"\
                    --output_folder="<path/where/the/output/is/saved"\
                    --token="<your Hugging Face token>"
```
### Personalized cut
```bash
python benchmark.py --personalized \
                    --model="meta-llama/Llama-2-7b-hf"\
                    --task="mmlu"\
                    --start_block=5\
                    --end_block=15\
                    --output_folder="<path/where/the/output/is/saved"\
                    --token="<your Hugging Face token>"
```
### Full model
```bash
python benchmark.py --full \
                    --model="meta-llama/Llama-2-7b-hf"\
                    --task="mmlu"\
                    --output_folder="<path/where/the/output/is/saved"\
                    --token="<your Hugging Face token>"
```

### Model names used in this work

 | Model  | ```--model``` input |
 |:------:|:---------:|
 |Llama 2 7B | meta-llama/Llama-2-7b-hf  |
 |Llama 2 13B | meta-llama/Llama-2-13b-hf |
 |Llama 2 70B | meta-llama/Llama-2-70b-hf |
 |Llama 3 8B  | meta-llama/Meta-Llama-3-8B |
 |Llama 3 70B | meta-llama/Meta-Llama-3-70B |
 |Pythia 6.9B | EleutherAI/pythia-6.9b-deduped |
 |Mistral 7B  | mistralai/Mistral-7B-v0.1|