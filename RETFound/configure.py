from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class RETFoundArgs:
    # ---- Core training
    batch_size: int = 128
    epochs: int = 50
    accum_iter: int = 1

    # ---- Model parameters
    model: str = "vit_large_patch16"
    model_arch: str = "dinov3_vits16"
    input_size: int = 256
    drop_path: float = 0.2
    global_pool: bool = True  # default set by parser.set_defaults

    # ---- Optimizer parameters
    clip_grad: Optional[float] = None
    weight_decay: float = 0.05
    lr: Optional[float] = None
    blr: float = 5e-3
    layer_decay: float = 0.65
    min_lr: float = 1e-6
    warmup_epochs: int = 10

    # ---- Augmentation
    color_jitter: Optional[float] = None
    aa: str = "rand-m9-mstd0.5-inc1"
    smoothing: float = 0.1

    # ---- Random erase
    reprob: float = 0.25
    remode: str = "pixel"
    recount: int = 1
    resplit: bool = False

    # ---- Mixup/Cutmix
    mixup: float = 0.0
    cutmix: float = 0.0
    cutmix_minmax: Optional[List[float]] = None
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    mixup_mode: str = "batch"

    # ---- Finetuning & adaptation
    finetune: str = ""
    task: str = ""
    adaptation: str = "finetune"  # choices=["finetune", "lp"]

    # ---- Dataset & paths
    data_path: Path = "./data/"
    nb_classes: int = 8
    output_dir: Path = "./output_dir"
    log_dir: Path = "./output_logs"

    # >>> NEW: training data efficiency <<<
    dataratio: str = "1.0"
    stratified: bool = False

    # ---- Runtime
    device: str = "cuda"
    seed: int = 0
    resume: str = ""
    start_epoch: int = 0
    eval: bool = False
    dist_eval: bool = False
    num_workers: int = 10
    pin_mem: bool = True  # default set by parser.set_defaults

    # ---- Distributed
    world_size: int = 1
    local_rank: int = -1
    dist_on_itp: bool = False
    dist_url: str = "env://"

    # ---- Misc
    savemodel: bool = True
    norm: str = "IMAGENET"
    enhance: bool = False
    datasets_seed: int = 2026

    @classmethod
    def from_namespace(cls, ns: object) -> "RETFoundArgs":
        """Create RETFoundArgs from an argparse.Namespace-like object."""
        # Convert Namespace to dict safely (handles missing optional fields)
        data = {}
        for field in cls.__dataclass_fields__.values():  # type: ignore[attr-defined]
            name = field.name
            if hasattr(ns, name):
                data[name] = getattr(ns, name)
        return cls(**data)



