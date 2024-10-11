from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Sequence, Dict
import torch
from dataclasses import dataclass
from typing import Sequence, Dict

IGNORE_INDEX = -100


def get_dataloader(
    dataset,
    batch_size,
    pad_token_id=None,
    max_seq_len=2048,
    world_size=1,
    sampler=None,
    shuffle=False,
    drop_last=True,
    num_processes=1,
):
    if pad_token_id is None:
        # we are using the image encoder
        collate_fn = DataCollatorForViT()
    else:
        # we are using the text encoder
        collate_fn = DataCollatorForCausalLM(
            pad_token_id=pad_token_id, max_seq_len=max_seq_len
        )

    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=drop_last)
        shuffle = False

    dataloader = DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_processes,
        sampler=sampler,
        pin_memory=True,
    )

    return dataloader


@dataclass
class DataCollatorForCausalLM:
    """Collate examples for supervised fine-tuning."""

    pad_token_id: int
    max_seq_len: int

    # check if we can set padding value in labels == eos_token_id_directly (as the attention mask should take into account the gradient masking)

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # in the structure of open-instruct the instances are already tensors, and already take into account max_seq_len

        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        attention_mask = [instance["attention_mask"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


@dataclass
class DataCollatorForViT:
    """Collate examples for supervised fine-tuning."""

    def __call__(self, batch: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # in the structure of open-instruct the instances are already tensors, and already take into account max_seq_len
        assert isinstance(batch[0], tuple)

        images = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        attention_mask = None

        return dict(input_ids=images, labels=labels, attention_mask=attention_mask)
