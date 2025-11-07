import math

import torch
from torch.optim.lr_scheduler import LambdaLR

# The following schedulers are based on https://github.com/facebookresearch/ijepa/blob/main/src/utils/schedulers.py

class WarmupCosineSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, start_lr, ref_lr, T_max, final_lr=0.0):
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.T_max = T_max - warmup_steps
        self.final_lr = final_lr

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup phase
                progress = float(current_step) / float(max(1, warmup_steps))
                return (self.start_lr + progress * (self.ref_lr - self.start_lr)) / self.ref_lr
            else:
                # Cosine annealing phase
                progress = float(current_step - warmup_steps) / float(max(1, self.T_max))
                cosine_lr = max(
                    self.final_lr,
                    self.final_lr
                    + (self.ref_lr - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
                )
                return cosine_lr / self.ref_lr

        super().__init__(optimizer, lr_lambda)

class CosineWDSchedule:
    def __init__(self, optimizer, ref_wd, T_max, final_wd=0.0, current_step=0):
        """
        Implements a cosine schedule for weight decay.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer.
            ref_wd (float): Initial weight decay.
            T_max (int): Total number of training steps.
            final_wd (float, optional): Minimum weight decay. Defaults to 0.0.
        """
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = current_step  # Tracks current step

    def step(self):
        """Updates weight decay according to the cosine schedule."""
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        # Ensure WD stays within bounds
        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        # Apply weight decay update to optimizer
        for group in self.optimizer.param_groups:
            if "WD_exclude" not in group or not group["WD_exclude"]:
                group["weight_decay"] = new_wd  # Correctly apply weight decay

    def state_dict(self):
        """Returns the scheduler state for checkpointing."""
        return {
            "step": self._step,
        }

    def load_state_dict(self, state_dict):
        """Loads the scheduler state from a checkpoint."""
        self._step = state_dict.get("step", 0)

    def get_last_wd(self):
        """Returns the most recently applied weight decay value."""
        return self.optimizer.param_groups[0]["weight_decay"]
