from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass
class TrainConfig:
    train_log: int = 2
    val_log: int = 20
    save_interval: int = 20
    eval_iters: int = 100
    eval_max_new_tokens: int = 100
    num_train_samples: Optional[int] = None

    # Hyperparameters
    max_seq_len: int = 4096  # for llama2 this can be extended to 4096
    epochs: int = 1
    optimizer: Optional[str] = None
    lr_scheduler_type: Optional[str] = None
    learning_rate: float = 3e-4
    batch_size: int = 128
    micro_batch_size: int = 1
    micro_batch_size_val: int = 1
    max_iters: int = 50000  # train dataset size
    weight_decay: float = 0.01
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_query: bool = True
    lora_key: bool = True  # False
    lora_value: bool = True
    lora_projection: bool = True  # False
    lora_mlp: bool = True  # False
    lora_head: bool = True  # False
    warmup_steps: Optional[int] = None
    warmup_ratio: Optional[float] = None

    # SETUP
    use_all_tokens: Optional[bool] = False
    use_last_token: Optional[bool] = False
    use_all_samples: Optional[bool] = False
    save_distances: Optional[bool] = False
    target_layer: Optional[str] = "norm1"
    layer_interval: Optional[int] = 1
    model_name: Optional[str] = None
    compute_id: bool = False
    id_samples: Optional[int] = None
    # hardware
    num_processes: int = 6
    # files
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_info: Optional[str] = None
    data_dir: Path = Path("data/alpaca")
    mmlu_dir: Path = Path("data/mmlu")
    checkpoint_dir: Path = Path("input/must/be/path")
    tokenizer_dir: Optional[Path] = None
    out_dir: Path = Path("./results")
    out_filename: str = ""
    precision: Optional[str] = None
    quantize: Optional[
        Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8-training"]
    ] = None


@dataclass
class open_instruct:
    learning_rate: float = 2e-5  # 1e-5 for 30/65B/70B #1e-4 qlora
    batch_size: int = 128  # (tulu v2)
    epochs: int = 2
    weight_decay: float = 0
    optimizer: str = "sgd"
    lr_scheduler_type: str = "linear"
    lora_r: int = 64  # qlora
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_query: bool = True
    lora_key: bool = True  # False
    lora_value: bool = True
    lora_projection: bool = True  # False
    lora_mlp: bool = True  # False
    lora_head: bool = True  # False
    warmup_ratio: float = 0.03


@dataclass
class lit_gpt:
    learning_rate: float = 2e-5  # 1e-5 for 30/65B/70B
    batch_size: int = 16  # 128 (16 lima)
    weight_decay: float = 0.01  #
    epochs: int = 2
    lora_r: int = 256
    lora_alpha: int = 512
    optimizer: str = "adamw"  # also SGD + momentum
    lr_scheduler_type: str = "cosine"
    lora_dropout: float = 0.05
    lora_query: bool = True
    lora_key: bool = True  # False
    lora_value: bool = True
    lora_projection: bool = True  # False
    lora_mlp: bool = True  # False
    lora_head: bool = True  # False
    warmup_ratio: float = 0.03


@dataclass
class alpaca:
    epochs: int = 3  # 5 for 13B
    max_seq_length: int = 1024  # at least 512
    learning_rate: float = 2e-5  # 1e-5 for 13B
    weight_decay: float = 0
    lr_lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    batch_size: int = 128


@dataclass
class lima:
    batch_size: int = 64
    epochs: int = 5  # actually 15
    max_seq_len: int = 2048  # actually 2048
    optimizer: str = "adamw"
    adamb1: float = 0.9
    adamb2: float = 0.95
    weight_decay: float = 0.1
    warmup_steps: int = 0
    learning_rate: float = 1e-5
    final_learning_rate: float = 1e-6
    lr_scheduler_type: str = "linear"
    residual_dropout: Optional[float] = None  # actually increasing from 0 to 0.2 in out
