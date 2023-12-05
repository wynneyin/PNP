import math
from typing import Any

import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from .. import functional as F
from .. import init
from .module import Module
from .lazy import LazyModuleMixin


__all__ = [
    'Linear'
]


class NTT_zkp(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features', 'windows_size', 'gpu_id',
                     'is_intt', 'partial_twiddles', 'radix_twiddles', 'radix_middles',
                     'partial_group_gen_powers', 'domain_size_inverse']
    
    windows_size: int
    gpu_id: int
    is_intt: bool
    partial_twiddles: Tensor
    radix_twiddles: Tensor
    radix_middles: Tensor
    partial_group_gen_powers: Tensor
    domain_size_inverse: Tensor


    def __init__(self, domain_size: int, gpu_id: int, is_intt: bool, 
                 partial_twiddles: Tensor, radix_twiddles: Tensor, radix_middles: Tensor,
                 partial_group_gen_powers: Tensor, domain_size_inverse: Tensor) -> None:
        super().__init__()
        windows_size = 2**14 #LG_WINDOW_SIZE ((MAX_LG_DOMAIN_SIZE + 1) / 2) 
        size_partial_twiddles = (windows_size, 4)
        size_radix_twiddles = (64 + 128 + 256 + 512 + 32, 4)
        size_radix_middles = (64*64 + 4096*64 + 128*128 + 256*256 + 512*512, 4)
        size_partial_group_gen_powers = (windows_size, 4)
        size_domain_size_inverse = (domain_size+1, 4) 
        partial_twiddles = torch.zeros(size_partial_twiddles, dtype = torch.BLS12_381_Fq_G1_Mont)
        radix_twiddles = torch.zeros(size_radix_twiddles, dtype = torch.BLS12_381_Fq_G1_Mont)
        radix_middles = torch.zeros(size_radix_middles, dtype = torch.BLS12_381_Fq_G1_Mont)
        partial_group_gen_powers = torch.zeros(size_partial_group_gen_powers, dtype = torch.BLS12_381_Fq_G1_Mont)
        domain_size_inverse = torch.zeros(size_domain_size_inverse, dtype = torch.uint32_t)
        self.partial_twiddles = partial_twiddles
        self.radix_twiddles = radix_twiddles
        self.radix_middles = radix_middles
        self.partial_group_gen_powers = partial_group_gen_powers
        self.domain_size_inverse = domain_size_inverse
        Parmas = torch.tensor_list([self.partial_twiddles, self.radix_twiddles, self.radix_middles,
                                    self.partial_group_gen_powers, self.domain_size_inverse])
        Parmas = torch.params_zkp(Parmas, gpu_id, is_intt, windows_size)

    def forward(self, input: Tensor, is_intt: bool, is_coset: bool) -> Tensor:
        if is_intt & is_coset:
            output = torch.intt_coset_zkp(input, self.partial_twiddles, self.radix_twiddles,
                              self.radix_middles, self.partial_group_gen_powers,
                              self.domain_size_inverse)
        elif is_intt:
            output = torch.intt_zkp(input, self.partial_twiddles, self.radix_twiddles,
                              self.radix_middles, self.partial_group_gen_powers,
                              self.domain_size_inverse)
        elif is_coset:
            output = torch.ntt_coset_zkp(input, self.partial_twiddles, self.radix_twiddles,
                              self.radix_middles, self.partial_group_gen_powers,
                              self.domain_size_inverse)
        else:
            output = torch.ntt_zkp(input, self.partial_twiddles, self.radix_twiddles,
                              self.radix_middles, self.partial_group_gen_powers,
                              self.domain_size_inverse)
        return output


    
