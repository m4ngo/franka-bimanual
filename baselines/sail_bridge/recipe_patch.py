"""Make SAIL's robomimic honour baselines/diffusion_recipe.py. Runs in .venv-sail.

    from baselines.sail_bridge import recipe_patch; recipe_patch.install()

before any robomimic config is built, i.e. before training starts and before
`FileUtils.policy_from_checkpoint`. Upstream hardcodes what the recipe changes:

  * `DiffusionPolicyUNet._create_networks` builds `ConditionalUnet1D(input_dim,
    global_cond_dim)` with its default widths, ignoring `algo.unet.*`, which the
    config declares; and `EMAModel(power=...)` without warmup, so diffusers'
    (1 + n) / (10 + n) runs, not the power schedule.
  * the optimizer is Adam or AdamW at torch's default betas and eps.
  * the LR schedule is stepped once per epoch, from `on_epoch_end`.

Every key added here defaults to upstream's behaviour, so a config or checkpoint
that does not set them builds exactly what it did before. The submodule itself
is untouched.

On `train.load_ckpt`, robomimic rebuilds the model AND its optimizer from the
checkpoint's stored config, so the resumed run's own schedule (its start_step)
would never reach the scheduler; robomimic_train.py passes that block to
install() as `lr_override`.
"""

from __future__ import annotations

import functools

_INSTALLED = False
_LR_OVERRIDE: dict = {}

# key path under config.algo -> upstream behaviour
_DEFAULTS = {
    ("optim_params", "policy", "optimizer_type"): "adam",
    ("optim_params", "policy", "betas"): [0.9, 0.999],
    ("optim_params", "policy", "eps"): 1e-8,
    # null: robomimic's own epoch_schedule handling.
    ("optim_params", "policy", "learning_rate", "step_scheduler"): None,
    # robomimic reads this with .get(..., 10000) for constant_with_warmup.
    ("optim_params", "policy", "learning_rate", "num_warmup_steps"): 10000,
    ("optim_params", "policy", "learning_rate", "num_training_steps"): None,
    ("optim_params", "policy", "learning_rate", "start_step"): 0,
    ("ema", "use_warmup"): False,
    ("ema", "inv_gamma"): 1.0,
    ("ema", "min_decay"): 0.0,
    ("ema", "max_decay"): 0.9999,
    ("ema", "update_after_step"): 0,
}


def _register_keys() -> None:
    from robomimic.config.diffusion_policy_config import DiffusionPolicyConfig

    original = DiffusionPolicyConfig.algo_config

    @functools.wraps(original)
    def algo_config(self):
        original(self)
        for path, value in _DEFAULTS.items():
            node = self.algo
            for key in path[:-1]:
                node = getattr(node, key)
            setattr(node, path[-1], value)

    DiffusionPolicyConfig.algo_config = algo_config


def _patch_networks() -> None:
    import robomimic.algo.diffusion_policy as dp

    original = dp.DiffusionPolicyUNet._create_networks

    @functools.wraps(original)
    def _create_networks(self):
        # .get: a config loaded from a checkpoint is the stored dict, without
        # algo_config's defaults, and an older one lacks the EMA keys.
        unet, ema = self.algo_config.unet, self.algo_config.ema
        unet_cls, ema_cls = dp.ConditionalUnet1D, dp.EMAModel
        dp.ConditionalUnet1D = functools.partial(
            unet_cls,
            diffusion_step_embed_dim=int(unet.get("diffusion_step_embed_dim", 256)),
            down_dims=list(unet.get("down_dims", [256, 512, 1024])),
            kernel_size=int(unet.get("kernel_size", 5)),
            n_groups=int(unet.get("n_groups", 8)),
        )
        ema_kwargs = {key: ema.get(name, _DEFAULTS[("ema", name)]) for key, name in (
            ("decay", "max_decay"), ("min_decay", "min_decay"),
            ("update_after_step", "update_after_step"), ("use_ema_warmup", "use_warmup"),
            ("inv_gamma", "inv_gamma"))}
        dp.EMAModel = functools.partial(ema_cls, **ema_kwargs)
        try:
            return original(self)
        finally:
            dp.ConditionalUnet1D, dp.EMAModel = unet_cls, ema_cls

    dp.DiffusionPolicyUNet._create_networks = _create_networks


def _patch_optim() -> None:
    import torch
    import robomimic.utils.torch_utils as TorchUtils

    original_sched = TorchUtils.lr_scheduler_from_optim_params
    original_backprop = TorchUtils.backprop_for_loss

    def optimizer_from_optim_params(net_optim_params, net):
        kind = net_optim_params.get("optimizer_type", "adam")
        cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[kind]
        return cls(
            params=net.parameters(),
            lr=net_optim_params["learning_rate"]["initial"],
            betas=tuple(net_optim_params.get("betas", (0.9, 0.999))),
            eps=float(net_optim_params.get("eps", 1e-8)),
            weight_decay=net_optim_params["regularization"]["L2"],
        )

    def lr_scheduler_from_optim_params(net_optim_params, net, optimizer):
        lr = {**net_optim_params["learning_rate"], **_LR_OVERRIDE}
        name = lr.get("step_scheduler", None)
        if name is None:
            return original_sched(net_optim_params, net, optimizer)
        from diffusers.optimization import get_scheduler

        if lr.get("num_training_steps", None) is None:
            raise ValueError("learning_rate.step_scheduler needs learning_rate.num_training_steps")
        start = int(lr.get("start_step", 0))
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        optimizer._recipe_step_scheduler = get_scheduler(
            name, optimizer, num_warmup_steps=int(lr["num_warmup_steps"]),
            num_training_steps=int(lr["num_training_steps"]), last_epoch=start - 1)
        # None keeps on_epoch_end from stepping it a second time.
        return None

    def backprop_for_loss(net, optim, loss, max_grad_norm=None, retain_graph=False):
        grad_norms = original_backprop(net, optim, loss, max_grad_norm=max_grad_norm,
                                       retain_graph=retain_graph)
        sched = getattr(optim, "_recipe_step_scheduler", None)
        if sched is not None:
            sched.step()
        return grad_norms

    TorchUtils.optimizer_from_optim_params = optimizer_from_optim_params
    TorchUtils.lr_scheduler_from_optim_params = lr_scheduler_from_optim_params
    TorchUtils.backprop_for_loss = backprop_for_loss


def install(lr_override: dict | None = None) -> None:
    """`lr_override`: learning_rate keys that win over the model's own config."""
    global _INSTALLED
    _LR_OVERRIDE.clear()
    _LR_OVERRIDE.update(lr_override or {})
    if _INSTALLED:
        return
    _register_keys()
    _patch_networks()
    _patch_optim()
    _INSTALLED = True
