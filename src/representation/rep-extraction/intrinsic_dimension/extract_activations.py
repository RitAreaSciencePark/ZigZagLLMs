import time
import torch
from collections import defaultdict
import sys

# ***************************************************


class extract_activations:
    def __init__(
        self,
        model,
        model_name,
        dataloader,
        target_layers,
        embdim,
        dtypes,
        nsamples,
        use_last_token=False,
        print_every=100,
    ):
        self.model = model
        self.model_name = model_name
        # embedding size
        self.embdim = embdim
        # number of samples to collect (e.g. 10k)
        self.nsamples = nsamples
        # whether to compute the id on the last token /class_token
        self.use_last_token = use_last_token
        self.print_every = print_every

        self.tot_tokens = 0
        self.global_batch_size = dataloader.batch_size
        self.nbatches = len(dataloader)

        self.init_hidden_states(target_layers, dtypes=dtypes)
        self.init_hooks(target_layers)

    def init_hidden_states(self, target_layers, dtypes):
        # dict containing the representations extracted in a sigle forward pass
        self.hidden_states_tmp = defaultdict(lambda: None)

        # dict storing the all the representations
        self.hidden_states = {}
        for name in target_layers:
            self.hidden_states[name] = torch.zeros(
                (self.nsamples, self.embdim[name]), dtype=dtypes[name]
            )

    def init_hooks(self, target_layers):
        for name, module in self.model.named_modules():
            if name in target_layers:
                module.register_forward_hook(
                    self._get_hook(name, self.hidden_states_tmp)
                )

    def _get_hook(self, name, hidden_states):
        def hook_fn(module, input, output):
            hidden_states[name] = output.cpu()

        return hook_fn

    def _update_activation_dict_text(self, mask, is_last_batch):
        seq_len = torch.sum(mask, dim=1)
        mask = mask.unsqueeze(-1)

        for _, (layer, activations) in enumerate(self.hidden_states_tmp.items()):
            if self.use_last_token:
                batch_size = seq_len.shape[0]
                act_tmp = activations[
                    torch.arange(batch_size), torch.tensor(seq_len) - 1
                ]
            else:
                denom = torch.sum(mask, dim=1)  # batch x 1
                # act_tmp -> batch x seq_len x embed
                # mask -> batch x seq_len x 1
                # denom -> batch x 1
                act_tmp = torch.sum(activations * mask, dim=1) / denom

            num_current_tokens = act_tmp.shape[0]
            if is_last_batch:
                self.hidden_states[layer][self.tot_tokens :] = act_tmp
            else:
                self.hidden_states[layer][
                    self.tot_tokens : self.tot_tokens + num_current_tokens
                ] = act_tmp

        self.tot_tokens += num_current_tokens

    def _update_activation_dict_img(self, is_last_batch):

        for _, (layer, activations) in enumerate(self.hidden_states_tmp.items()):
            if self.use_last_token:
                # just use classification token
                act_tmp = activations[:, 0, :]
            else:
                # remove first_classification token and average over the sequence
                act_tmp = torch.mean(
                    activations[:, 1:, :],
                    dim=1,
                )

            num_current_tokens = act_tmp.shape[0]
            if is_last_batch:
                self.hidden_states[layer][self.tot_tokens :] = act_tmp
            else:
                self.hidden_states[layer][
                    self.tot_tokens : self.tot_tokens + num_current_tokens
                ] = act_tmp

        self.tot_tokens += num_current_tokens

    @torch.no_grad()
    def extract(self, dataloader):
        start = time.time()
        is_last_batch = False
        for i, batch in enumerate(dataloader):

            if (i + 1) == self.nbatches:
                is_last_batch = True

            if batch["attention_mask"] is not None:
                for key in batch.keys():
                    batch[key] = batch[key].to("cuda")
                    
                _ = self.model(**batch, use_cache=False)
                """
                if self.model_name.startswith("llama"):
                    _ = self.model(**batch, use_cache=False)
                    #print("model is llama")
                elif self.model_name.startswith("deberta"):
                    _ = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                """
                mask = batch["attention_mask"] != 0
                #print(f"mask before text {mask}")
                self._update_activation_dict_text(mask.cpu(), is_last_batch)
                #print(f"finished calling update text - {self.tot_tokens}")
            else:
                inputs = batch["input_ids"].to("cuda")
                _ = self.model(inputs)
                self._update_activation_dict_img(is_last_batch)

            if (i + 1) % (self.print_every // self.global_batch_size) == 0:
                end = time.time()
                print(
                    f"{(i+1)*self.global_batch_size/1000}k data, \
                    batch {i+1}/{self.nbatches}, \
                    tot_time: {(end-start)/60: .3f}min, "
                )
                sys.stdout.flush()
