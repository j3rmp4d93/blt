# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
import os
import re
from omegaconf import OmegaConf
import torch

from bytelatent.transformer import LMTransformer, LMTransformerArgs


def load_entropy_model(entropy_model_checkpoint_dir, state_dict_path, device="cpu"):
    with open(os.path.join(entropy_model_checkpoint_dir, "params.json")) as fr:
        reloaded = json.loads(fr.read())

    torch.set_default_dtype(torch.bfloat16)
    model_params = reloaded["model"]
    entropy_model = LMTransformer(
        LMTransformerArgs(
            dim=model_params["dim"],
            n_layers=model_params["n_layers"],
            n_heads=model_params["n_heads"],
            max_seqlen=model_params["max_length"],
            ffn_dim_multiplier=model_params["ffn_dim_multiplier"],
            vocab_size=model_params["vocab_size"],
        )
    )

    entropy_model.load_state_dict(
        torch.load(state_dict_path, map_location=device), strict=False
    )
    entropy_model.to(device)
    entropy_model = entropy_model.eval()
    # no grads for the model:
    for param in entropy_model.parameters():
        param.requires_grad = False
    return entropy_model

def main():
    from bytelatent.args import TrainArgs_entropy#to prevent circular import
    from bytelatent.train import train
    """
    The command line interface here uses OmegaConf https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments
    This accepts arguments as a dot list
    So if the dataclass looks like

    @dataclass
    class DummyArgs:
        name: str
        model: LMTransformerArgsgs

    @dataclass
    class LMTransformerArgsgs:
        dim: int

    Then you can pass model.dim=32 to change values in LMTransformerArgsgs
    or just name=tictac for top level attributes.

    The behavior here is as follows:
    1. We instantiate TrainArgs with its default values
    2. We override those default values with the ones in the provided config file
    3. We override the result with the additional arguments provided through command line

    For example, if the config is the following

    model:
        dim: 128
        n_layers: 4

    and you call train.py with train.py model.dim=64

    Then the final TrainArgs will have

    model:
        dim: 64
        n_layers: 4

    Plus all the default values in TrainArgs dataclass.
    """
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_args.config

    default_cfg = OmegaConf.create(TrainArgs_entropy().model_dump())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    train_args = TrainArgs_entropy.model_validate(cfg)
    train(train_args, LMTransformer)

if __name__ == "__main__":
    main()
