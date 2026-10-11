import math
from typing import Optional


class AlphaScheduler:
    """
    Alpha schedule evaluated once per cycle row.

    Modes:
      schedule: fixed | linear | cosine
      step_mode: epoch | total
      reverse: if True, run max->min instead of min->max
    """

    def __init__(
        self,
        alpha_min: float = 0.2,
        alpha_max: float = 2.0,
        schedule: str = "fixed",
        reverse: bool = False,
        step_mode: str = "epoch",
        warmup_steps: int = 0,
        num_classes: int = 1000,
        cycles_per_epoch: int = 1,
        total_epochs: Optional[int] = None,
        total_steps: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        if schedule not in {"fixed", "linear", "cosine"} or step_mode not in {
            "epoch",
            "total",
        }:
            raise ValueError("Unknown alpha schedule or step mode")
        if alpha_min <= 0 or alpha_max < alpha_min:
            raise ValueError("Require 0 < alpha_min <= alpha_max")
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.schedule = schedule.lower()
        self.reverse = bool(reverse)
        self.step_mode = step_mode.lower()
        self.cycles_per_epoch = max(1, int(cycles_per_epoch))

        self.warmup_cycles = 0
        if warmup_steps > 0 and batch_size and num_classes > 0:
            self.warmup_cycles = int(
                math.ceil((warmup_steps * batch_size) / num_classes)
            )

        self.total_global_cycles = 1
        if self.schedule != "fixed":
            if total_epochs is not None:
                self.total_global_cycles = int(total_epochs) * self.cycles_per_epoch
            elif total_steps is not None and batch_size and num_classes > 0:
                self.total_global_cycles = int(
                    math.ceil((total_steps * batch_size) / num_classes)
                )

        self.scope_cycles = (
            self.cycles_per_epoch
            if self.step_mode == "epoch"
            else self.total_global_cycles
        )

    def get_alpha(self, epoch: int, cycle_idx_in_epoch: int) -> float:
        if self.schedule == "fixed":
            return self.alpha_min

        global_idx = epoch * self.cycles_per_epoch + cycle_idx_in_epoch

        if global_idx < self.warmup_cycles:
            return self.alpha_max if self.reverse else self.alpha_min

        if self.step_mode == "epoch":
            p = cycle_idx_in_epoch / max(1, self.scope_cycles - 1)
        else:
            eff_curr = global_idx - self.warmup_cycles
            eff_tot = max(1, self.total_global_cycles - self.warmup_cycles - 1)
            p = eff_curr / eff_tot

        p = float(max(0.0, min(1.0, p)))
        if self.schedule == "cosine":
            p = 0.5 * (1.0 - math.cos(math.pi * p))

        span = self.alpha_max - self.alpha_min
        return (
            (self.alpha_max - p * span) if self.reverse else (self.alpha_min + p * span)
        )


class KScheduler:
    """
    Epoch-level integer K schedule.

    K is updated once per epoch and progresses over total epochs.
    Modes: fixed | linear | cosine

    Optional cooldown:
      - if ``cooldown_epochs`` is set (> 0), the last
        ``cooldown_epochs`` epochs force ``k = 1``.
    """

    def __init__(
        self,
        k_min: int = 4,
        k_max: int = 4,
        schedule: str = "fixed",
        reverse: bool = False,
        warmup_epochs: int = 0,
        total_epochs: Optional[int] = None,
        cooldown_epochs: Optional[int] = None,
    ) -> None:
        if schedule not in {"fixed", "linear", "cosine"}:
            raise ValueError("Unknown K schedule")
        self.k_min = int(k_min)
        self.k_max = int(k_max)
        if self.k_min < 1 or self.k_max < 1:
            raise ValueError("k_min and k_max must be >= 1")
        if self.k_max < self.k_min:
            raise ValueError(
                f"k_max must be >= k_min (got {self.k_max} < {self.k_min})"
            )

        self.schedule = str(schedule).lower()
        self.reverse = bool(reverse)
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.total_epochs = max(1, int(total_epochs) if total_epochs is not None else 1)

        if cooldown_epochs is None:
            self.cooldown_epochs: Optional[int] = None
        else:
            cooldown = int(cooldown_epochs)
            if cooldown < 0:
                raise ValueError("cooldown_epochs must be >= 0")
            self.cooldown_epochs = cooldown

    def get_k(self, epoch: int) -> int:
        epoch = int(epoch)

        if self.cooldown_epochs is not None and self.cooldown_epochs > 0:
            cooldown_start = max(0, self.total_epochs - self.cooldown_epochs)
            if epoch >= cooldown_start:
                return 1

        if self.schedule == "fixed" or self.k_min == self.k_max:
            return self.k_min

        if epoch < self.warmup_epochs:
            return self.k_max if self.reverse else self.k_min

        eff_curr = epoch - self.warmup_epochs
        eff_tot = max(1, self.total_epochs - self.warmup_epochs - 1)
        p = float(max(0.0, min(1.0, eff_curr / eff_tot)))

        if self.schedule == "cosine":
            p = 0.5 * (1.0 - math.cos(math.pi * p))

        span = float(self.k_max - self.k_min)
        kf = (self.k_max - p * span) if self.reverse else (self.k_min + p * span)
        k = int(math.floor(kf + 0.5))
        return max(self.k_min, min(self.k_max, k))
