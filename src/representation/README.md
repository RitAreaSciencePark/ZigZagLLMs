## Usage of the representation extraction code

To extract the representations of the dataset used in the paper you have to access the models, in this case we use the versions accessible on Huggin Face Hub. 

For the parameter ```--model``` use the string used to call the model from Hugging Face. In our case they are

 | Model  | ```--model``` input |
 |:------:|:---------:|
 |Llama 2 7B | meta-llama/Llama-2-7b-hf  |
 |Llama 2 13B | meta-llama/Llama-2-13b-hf |
 |Llama 2 70B | meta-llama/Llama-2-70b-hf |
 |Llama 3 8B  | meta-llama/Meta-Llama-3-8B |
 |Llama 3 70B | meta-llama/Meta-Llama-3-70B |
 |Pythia 6.9B | EleutherAI/pythia-6.9b-deduped |
 |Mistral 7B  | mistralai/Mistral-7B-v0.1|


## Extraction example
```bash
python rep-extraction/extract.py --model="meta-llama/Llama-2-7b-hf" \
                  --dataset="sst" \
                  --output_folder="rep-llama2-7b" \
                  --access_token="<your access token from HF>"

```