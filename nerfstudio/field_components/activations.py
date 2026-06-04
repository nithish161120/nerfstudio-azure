# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Special activation functions.
"""

from typing import TYPE_CHECKING

import torch
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

# torch.amp.custom_fwd/bwd were added in PyTorch 2.4.
# Fall back to torch.cuda.amp for older versions (e.g. 2.1.x used in the container).
try:
    from torch.amp import custom_fwd as _amp_fwd, custom_bwd as _amp_bwd
    _fwd_decorator = _amp_fwd(cast_inputs=torch.float32, device_type="cuda")
    _bwd_decorator = _amp_bwd(device_type="cuda")
except ImportError:
    from torch.cuda.amp import custom_fwd as _cuda_fwd, custom_bwd as _cuda_bwd
    _fwd_decorator = _cuda_fwd(cast_inputs=torch.float32)
    _bwd_decorator = _cuda_bwd


class _TruncExp(Function):
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @_fwd_decorator
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @_bwd_decorator
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


if TYPE_CHECKING:

    def trunc_exp(_: Float[Tensor, "*bs"], /) -> Float[Tensor, "*bs"]:
        """Same as torch.exp, but with the backward pass clipped to prevent vanishing/exploding
        gradients."""
        raise NotImplementedError()

else:
    trunc_exp = _TruncExp.apply
    """Same as torch.exp, but with the backward pass clipped to prevent vanishing/exploding
    gradients."""
