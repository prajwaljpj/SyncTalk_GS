#!/usr/bin/env python3


"""
Configuration loader for SyncTalk Streaming Inference.
Handles loading YAML config and merging with runtime parameters.
"""

import yaml
from argparse import Namespace
from pathlib import Path


def load_yaml_config(config_path="config/inference_config.yaml"):
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with configuration
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def create_opt(config, runtime_params):
    """
    Create opt Namespace from YAML config and runtime parameters.

    Args:
        config: Dictionary from YAML file
        runtime_params: Dictionary with runtime parameters like:
            - data_path
            - workspace
            - checkpoint
            - portrait
            - torso
            - W, H, fps

    Returns:
        Namespace object with all opt parameters
    """
    opt = Namespace(
        # ====================================================================
        # Runtime parameters (from arguments)
        # ====================================================================
        path=runtime_params["data_path"],
        workspace=runtime_params["workspace"],
        ckpt=runtime_params.get("checkpoint", "latest"),
        portrait=runtime_params.get("portrait", False),
        torso=runtime_params.get("torso", False),
        W=runtime_params.get("W", 512),
        H=runtime_params.get("H", 512),
        fps=runtime_params.get("fps", 25),
        # Runtime-only settings
        aud="",  # Will be set per inference
        head_ckpt="",  # No head checkpoint for streaming
        # ====================================================================
        # Model architecture (from YAML)
        # ====================================================================
        asr_model=config["model"]["asr_model"],
        emb=config["model"]["emb"],
        att=config["model"]["att"],
        exp_eye=config["model"]["exp_eye"],
        ind_dim=config["model"]["ind_dim"],
        ind_num=config["model"]["ind_num"],
        ind_dim_torso=config["model"]["ind_dim_torso"],
        amb_dim=config["model"]["amb_dim"],
        # ====================================================================
        # Scene settings (from YAML)
        # ====================================================================
        bound=config["scene"]["bound"],
        scale=config["scene"]["scale"],
        offset=config["scene"]["offset"],
        torso_shrink=config["scene"]["torso_shrink"],
        color_space=config["scene"]["color_space"],
        # ====================================================================
        # Rendering settings (from YAML)
        # ====================================================================
        dt_gamma=config["rendering"]["dt_gamma"],
        min_near=config["rendering"]["min_near"],
        density_thresh=config["rendering"]["density_thresh"],
        density_thresh_torso=config["rendering"]["density_thresh_torso"],
        max_steps=config["rendering"]["max_steps"],
        num_steps=config["rendering"]["num_steps"],
        upsample_steps=config["rendering"]["upsample_steps"],
        max_ray_batch=config["rendering"]["max_ray_batch"],
        # ====================================================================
        # Performance (from YAML)
        # ====================================================================
        fp16=config["performance"]["fp16"],
        cuda_ray=config["performance"]["cuda_ray"],
        patch_size=config["performance"]["patch_size"],
        preload=config["performance"]["preload"],
        update_extra_interval=config["performance"]["update_extra_interval"],
        # ====================================================================
        # Face settings (from YAML)
        # ====================================================================
        au45=config["face"]["au45"],
        fix_eye=config["face"]["fix_eye"],
        smooth_eye=config["face"]["smooth_eye"],
        smooth_lips=config["face"]["smooth_lips"],
        bs_area=config["face"]["bs_area"],
        # ====================================================================
        # Background (from YAML)
        # ====================================================================
        bg_img=config["background"]["bg_img"],
        fbg=config["background"]["fbg"],
        # ====================================================================
        # Training settings (from YAML)
        # ====================================================================
        train_camera=config["training"]["train_camera"],
        finetune_lips=config["training"]["finetune_lips"],
        init_lips=config["training"]["init_lips"],
        warmup_step=config["training"]["warmup_step"],
        amb_aud_loss=config["training"]["amb_aud_loss"],
        amb_eye_loss=config["training"]["amb_eye_loss"],
        unc_loss=config["training"]["unc_loss"],
        lambda_amb=config["training"]["lambda_amb"],
        pyramid_loss=config["training"]["pyramid_loss"],
        # ====================================================================
        # Data settings (from YAML)
        # ====================================================================
        data_range=config["data"]["data_range"],
        part=config["data"]["part"],
        part2=config["data"]["part2"],
        # ====================================================================
        # Camera settings (from YAML)
        # ====================================================================
        smooth_path=config["camera"]["smooth_path"],
        smooth_path_window=config["camera"]["smooth_path_window"],
        radius=config["camera"]["radius"],
        fovy=config["camera"]["fovy"],
        # ====================================================================
        # Mode settings (from YAML)
        # ====================================================================
        test=config["mode"]["test"],
        test_train=config["mode"]["test_train"],
        gui=config["mode"]["gui"],
        asr=config["mode"]["asr"],
        max_spp=1,
    )

    return opt


def load_inference_config(
    data_path,
    workspace,
    checkpoint="latest",
    portrait=False,
    torso=False,
    W=512,
    H=512,
    fps=25,
    config_path="config/inference_config.yaml",
):
    """
    Convenience function to load config and create opt in one call.

    Args:
        data_path: Path to training data directory
        workspace: Path to trained model workspace
        checkpoint: Checkpoint name or 'latest'
        portrait: Enable portrait mode
        torso: Enable torso mode
        W, H: Output dimensions
        fps: Frames per second
        config_path: Path to YAML config file

    Returns:
        Namespace object with all opt parameters
    """
    # Load YAML config
    config = load_yaml_config(config_path)

    # Create runtime params dict
    runtime_params = {
        "data_path": data_path,
        "workspace": workspace,
        "checkpoint": checkpoint,
        "portrait": portrait,
        "torso": torso,
        "W": W,
        "H": H,
        "fps": fps,
    }

    # Create and return opt
    return create_opt(config, runtime_params)
