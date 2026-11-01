import math

import torch
from torch import nn


class _SurrogateSpike(torch.autograd.Function):

    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (voltage,) = ctx.saved_tensors
        slope = 25.0
        surrogate_grad = 1.0 / (1.0 + slope * voltage.abs()).square()
        return grad_output * surrogate_grad


class LIFCell(nn.Module):
    # один шаг независимых LIF-нейронов, membrane хранит состояние между шагами

    def __init__(self, *, dt_s: float, tau_m_s: float = 0.05, threshold: float = 1.0,):
        super().__init__()

        parameters = {"dt_s": dt_s, "tau_m_s": tau_m_s, "threshold": threshold}

        for name, value in parameters.items():
            
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} должен быть конечным положительным числом")

        self.dt_s = float(dt_s)
        self.tau_m_s = float(tau_m_s)

        self.register_buffer("beta", torch.tensor(math.exp(-dt_s / tau_m_s), dtype=torch.float32))
        self.register_buffer("threshold", torch.tensor(threshold, dtype=torch.float32))

    def forward(self, drive: torch.Tensor, membrane: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(drive, torch.Tensor):
            raise TypeError("drive должен быть тензором pytorch")

        if not drive.is_floating_point():
            raise TypeError("drive должен иметь вещественный тип")

        if membrane is None:
            membrane = torch.zeros_like(drive)
        elif (membrane.shape != drive.shape or membrane.device != drive.device or membrane.dtype != drive.dtype):
            raise ValueError( "membrane и drive должны иметь одинаковые форму, устройство и тип")

        voltage = self.beta * membrane + drive

        spikes = _SurrogateSpike.apply(voltage - self.threshold)

        # вычитаем порог, не проводя градиент через дискретный сброс.
        new_membrane = (voltage - spikes.detach() * self.threshold)

        return spikes, new_membrane


def main():
    neuron = LIFCell(dt_s=1.0 / 30.0, tau_m_s=0.05, threshold=1.0,)

    membrane = None
    drive = torch.tensor([0.6], dtype=torch.float32)

    with torch.no_grad():
        for step in range(3):
            spikes, membrane = neuron(drive, membrane)

            print(f"шаг={step} " f"спайк={int(spikes.item())}" f"мембрана={membrane.item():.4f}")


if __name__ == "__main__":
    main()