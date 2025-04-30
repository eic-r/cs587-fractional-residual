from math import gamma
from collections import deque
import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision.models import resnet18

# Fractional binomial coefficient
def binom(a, k):
    try:
        return gamma(a + 1) / (gamma(k + 1) * gamma(a - k + 1))
    except ValueError:
        return 0.0

class FractionalGradFunction(Function):
    @staticmethod
    def forward(ctx, input, alpha, history_buffer, h, clip_threshold):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        ctx.history_buffer = history_buffer
        ctx.h = h
        ctx.clip_threshold = clip_threshold
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        history_buffer = ctx.history_buffer
        h = ctx.h
        clip_threshold = ctx.clip_threshold

        if clip_threshold is not None:
            norm = grad_output.norm()
            if norm > clip_threshold:
                grad_output = grad_output * (clip_threshold / (norm + 1e-6))

        history_buffer.appendleft(grad_output.detach())
        if len(history_buffer) > 20:
            history_buffer.pop()

        coeffs = []
        for k in range(len(history_buffer)):
            try:
                coeffs.append(((-1)**k) * binom(alpha, k))
            except ValueError:
                coeffs.append(0.0)
        coeffs = torch.tensor(coeffs, device=grad_output.device)

        stacked = torch.stack(list(history_buffer))
        weighted = coeffs.view(-1, *([1] * (grad_output.ndim))) * stacked
        gl_grad = torch.sum(weighted, dim=0)

        gl_grad = gl_grad / (h ** alpha)
        return gl_grad, None, None, None, None


class FractionalWrapper(nn.Module):
    def __init__(self, alpha=0.9, h=1.0, max_history=20, clip_threshold=5.0):
        super().__init__()
        self.alpha = alpha
        self.h = h
        self.max_history = max_history
        self.clip_threshold = clip_threshold
        self.history_buffers = {}

    def wrap(self, name, module):
        def forward_hook(module, input, output):
            if name not in self.history_buffers:
                self.history_buffers[name] = deque(maxlen=self.max_history)
            output = FractionalGradFunction.apply(
                output, self.alpha, self.history_buffers[name], self.h, self.clip_threshold
            )
            return output
        module.register_forward_hook(forward_hook)


class FractionalResNet18(nn.Module):
    def __init__(self, alpha=0.9, h=1.0, max_history=20, num_classes=10):
        super().__init__()
        self.base = resnet18()
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

        self.fw = FractionalWrapper(alpha=alpha, h=h, max_history=max_history)
        # You can choose which layers to apply fractional gradients to
        self.fw.wrap('layer1', self.base.layer1)
        self.fw.wrap('layer2', self.base.layer2)
        self.fw.wrap('layer3', self.base.layer3)
        self.fw.wrap('layer4', self.base.layer4)

    def forward(self, x):
        return self.base(x)