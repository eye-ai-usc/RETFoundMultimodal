from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class RETFoundArgs:
    """Configuration for RETFound fine-tuning and evaluation.

    This dataclass mirrors the arguments defined in main_finetune.get_args_parser
    and can be used as a typed, IDE-friendly replacement for argparse results.

    Typical usage:
    - Parse CLI and get a ready-to-use instance via main_finetune.get_args_parser().
    - Or build from an argparse.Namespace using RETFoundArgs.from_namespace(..).

    Attributes:
        batch_size (int): Per-GPU batch size. Default: 128.
        epochs (int): Number of training epochs. Default: 50.
        accum_iter (int): Gradient accumulation steps. Default: 1.

        model (str): Model entry in models_vit.py. Default: "vit_large_patch16".
        model_arch (str): Backbone architecture key (e.g., dinov3_vits16). Default: "dinov3_vits16".
        input_size (int): Image input size. Default: 256.
        drop_path (float): Drop path rate. Default: 0.2.
        global_pool (bool): Use global pooling for classification. Default: True.

        clip_grad (float | None): Max gradient norm for clipping. Default: None.
        weight_decay (float): Weight decay for AdamW. Default: 0.05.
        lr (float | None): Absolute learning rate; if None, computed from blr and batch size. Default: None.
        blr (float): Base learning rate used to compute lr. Default: 5e-3.
        layer_decay (float): Layer-wise LR decay for ViT. Default: 0.65.
        min_lr (float): Lower bound for LR schedulers. Default: 1e-6.
        warmup_epochs (int): Number of warmup epochs. Default: 10.

        color_jitter (float | None): Color jitter strength. Default: None.
        aa (str): AutoAugment policy. Default: "rand-m9-mstd0.5-inc1".
        smoothing (float): Label smoothing value. Default: 0.1.

        reprob (float): Random erase probability. Default: 0.25.
        remode (str): Random erase mode. Default: "pixel".
        recount (int): Random erase count. Default: 1.
        resplit (bool): Do not random erase first augmentation split. Default: False.

        mixup (float): Mixup alpha (enabled if > 0). Default: 0.0.
        cutmix (float): CutMix alpha (enabled if > 0). Default: 0.0.
        cutmix_minmax (list[float] | None): Min/max CutMix ratio, if provided. Default: None.
        mixup_prob (float): Probability to apply Mixup/CutMix. Default: 1.0.
        mixup_switch_prob (float): Probability to switch to CutMix. Default: 0.5.
        mixup_mode (str): Mixup/CutMix application mode. Default: "batch".

        finetune (str): Checkpoint id/path for loading pre-trained weights. Default: "".
        task (str): Task identifier for logging/output grouping. Default: "".
        adaptation (str): Adaptation strategy: "finetune" or "lp" (linear probe). Default: "finetune".

        data_path (pathlib.Path): Dataset root path. Default: "./data/".
        nb_classes (int): Number of target classes. Default: 8.
        output_dir (pathlib.Path): Directory to save outputs and checkpoints. Default: "./output_dir".
        log_dir (pathlib.Path): Directory to save TensorBoard logs. Default: "./output_logs".

        dataratio (str): Training data ratio(s) for subsampling (e.g., "1.0" or "1.0,0.5"). Default: "1.0".
        stratified (bool): Use class-stratified subsampling (if supported by dataset builder). Default: False.

        device (str): PyTorch device string. Default: "cuda".
        seed (int): Random seed (offset by process rank). Default: 0.
        resume (str): Path to checkpoint for resuming full state. Default: "".
        start_epoch (int): Starting epoch index. Default: 0.
        eval (bool): If True, run evaluation only. Default: False.
        dist_eval (bool): If True, enable distributed evaluation. Default: False.
        num_workers (int): DataLoader workers. Default: 10.
        pin_mem (bool): Pin CPU memory for efficient GPU transfer. Default: True.

        world_size (int): Number of distributed processes. Default: 1.
        local_rank (int): Local rank for distributed training. Default: -1.
        dist_on_itp (bool): Enable distributed training on ITP. Default: False.
        dist_url (str): init_method URL for process group. Default: "env://".

        savemodel (bool): Save best model checkpoint during training. Default: True.
        norm (str): Normalization method. Default: "IMAGENET".
        enhance (bool): Use enhanced data, if supported. Default: False.
        datasets_seed (int): Dataset-related seed. Default: 2026.

    Note:
        During runtime, distributed initialization may add extra attributes such as
        args.rank, args.gpu, args.distributed, and args.dist_backend via util.misc.init_distributed_mode.
    """
    # ---- Core training
    batch_size: int = 128  # Batch size per GPU (default: `128`)
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



