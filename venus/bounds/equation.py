"""
# File: equation.py
# Top contributors (to current version):
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus  project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description:  Defines the bound equation (abstract) class.
"""

from __future__ import annotations
from distutils.command.config import config

import math
import torch
import numpy as np

from venus.network.node import Node, Gemm, Conv, Relu, MatMul, Add, Sub, Constant
from venus.common.configuration import Config
from venus.gemmpy.calculation_apis import *

torch.set_num_threads(1)

class Equation():
    """
    The Equation class.
    """

    def __init__(self, matrix: torch.Tensor, const: torch.Tensor, config: Config):
        """
        Arguments:

            matrix:
                2D matrix (of size nxm). Each row i represents node's i
                equation coefficients.
            const:
                Vector (of size n). Each row i represents node's i equation
                constant term.
        """
        self.matrix = matrix
        self.matrix_intv = None
        self.const = const
        self.const_intv = None
        self.config = config
        self.size = matrix.shape[0]
        
        if config.interval_in_bs is True and config.gpu_in_bs is True:
            if (self.matrix is not None) and (self.const is not None):
                self.matrix_intv = torch_float_array2pinterval_gpu_array(matrix)
        # if config.interval_in_bs is True and config.gpu_in_bs is False:
        #     self.matrix_intv = float_array2np_interval_cpu_array(matrix)
        #     self.const_intv = float_array2np_interval_cpu_array(const)
        # elif config.interval_in_bs is True and config.gpu_in_bs is True:
        #     self.matrix_intv = torch_float_array2pinterval_gpu_array(matrix)
        #     self.const_intv = torch_float_array2pinterval_gpu_array(const)

    def copy(self, matrix=None, const=None) -> Equation:
        return Equation(
            self.matrix.detach().clone(),
            self.const.detach().clone(),
            self.size
        )
 
    def _get_plus_matrix(self, force_float = False) -> torch.Tensor:
        """
        Clips the coeffs to be only positive.
        """
        
        if force_float is True:
            res = torch.clamp(self.matrix, 0, math.inf)
            return res
        if self.config.interval_in_bs is True and self.config.gpu_in_bs is False:
            res = torch.clamp(self.matrix, 0, math.inf)
            return float_array2np_interval_cpu_array(res)
        elif self.config.interval_in_bs is True and self.config.gpu_in_bs is True:
            return torch.clamp(self.matrix_intv, 0, math.inf)
        return torch.clamp(self.matrix, 0, math.inf)


    def _get_minus_matrix(self, keep_in_memory=True, force_float = False) -> torch.Tensor:
        """
        Clips the coeffs to be only negative.
        """
        
        if force_float is True:
            res = torch.clamp(self.matrix, -math.inf, 0)
            return res
        if self.config.interval_in_bs is True and self.config.gpu_in_bs is False:
            res = torch.clamp(self.matrix, -math.inf, 0)
            return float_array2np_interval_cpu_array(res)
        elif self.config.interval_in_bs is True and self.config.gpu_in_bs is True:
            return torch.clamp(self.matrix_intv, -math.inf, 0)
        return torch.clamp(self.matrix, -math.inf, 0)


    def concrete_values(self, lower, upper, bound: str) -> torch.Tensor:
        if bound == 'lower':
            return self.min_values(lower, upper)
        
        elif bound == 'upper':
            return self.max_values(lower, upper)
       
        else:
            raise ValueError(f'Bound {bound} is not recognised.')
        

    def max_values(self, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        """
        Computes the upper bounds of the equations.
    
        Arguments:

            lower:
                The lower bounds for the variables in the equation. 
            upper:
                The upper bounds for the variables in the equation.
        
        Returns: 

            The upper bounds of the equation.
        """
        return  self.interval_dot('upper', lower, upper)


    def min_values(self, lower, upper):
        """
        Computes the lower bounds of the equations.

        Arguments:

            lower:
                The lower bounds for the variables in the equation. 
            upper:
                The upper bounds for the variables in the equation.

        Returns:

            The lower bounds of the equations.
        """

        return self.interval_dot('lower', lower, upper)


    def interval_dot(self, bound: str, lower, upper) -> torch.Tensor:
        """
        Computes the interval dot product with either a matrix or an Equation.
        """
        if bound == 'upper':
            # delegate to numpy sound calculation
            if self.config.interval_in_bs is True and self.config.gpu_in_bs is False:
                return get_upper(
                    add(mat_mul(self._get_plus_matrix(), upper, True),
                        mat_mul(self._get_minus_matrix(), lower, True),
                        True)
                ).reshape(-1) + self.const
            # delegate to gemmc sound calculation API
            if self.config.interval_in_bs is True and self.config.gpu_in_bs is True:
                return get_upper(
                    add(mat_mul(self._get_plus_matrix(), upper, True),
                        mat_mul(self._get_minus_matrix(), lower, True),
                        True)
                ).reshape(-1) + self.const
            # non-interval cases can be calculated by torch
            return self._get_plus_matrix() @ upper + \
                self._get_minus_matrix() @ lower + \
                self.const

        elif bound == 'lower':
            # delegate to numpy sound calculation
            if self.config.interval_in_bs is True and self.config.gpu_in_bs is False:
                return get_lower(
                    add(mat_mul(self._get_plus_matrix(), lower, True),
                        mat_mul(self._get_minus_matrix(), upper, True),
                        True)
                ).reshape(-1) + self.const
            # delegate to gemmc sound calculation API
            if self.config.interval_in_bs is True and self.config.gpu_in_bs is True:
                return get_lower(
                    add(mat_mul(self._get_plus_matrix(), lower, True),
                        mat_mul(self._get_minus_matrix(), upper, True),
                        True)
                ).reshape(-1) + self.const
            # non-interval cases can be calculated by torch
            return  self._get_plus_matrix() @ lower + \
                self._get_minus_matrix() @ upper + \
                self.const

        else: 
            raise ValueError(f'Bound {bound} is not recognised.')


    def pool(self, in_shape, out_shape, pooling):
        """
        Derives the pooling indices of the equations.

        Arguments:
            
            in_shape: tuple of the shape of the equations.

            out_shape: tuple the shape of the equations after pooling.

            pooling: tuple of the pooling size.
        
        Returns:

            List where each item i in the list is a list of indices of the i-th
            pooling neighbourhood.
        """
            
        m,n,_ = in_shape
        M,N,K =  out_shape
        p = pooling[0]
        m_row_extent = M*N*K
        # get Starting block indices
        start_idx = np.arange(p)[:,None]*n*K + np.arange(0,p*K,K)
        # get offsetted indices across the height and width of input 
        offset_idx = np.arange(M)[:,None]*n*K*p + torch.Tensor([i*p*K + np.arange(K) for i in  range(N)]).flatten()
        # get all actual indices 
        idx = start_idx.ravel()[:,None] + offset_idx.ravel()
        return [idx[i,:] for i in idx.shape[0]]

    def split_dot(self, a, b):
        size = a.shape[0] 
        c = np.empty(shape=(size, b.shape[1]))
        # est = sys.getsizeof(m) + sys.getsizeof(o) + 2*sys.getsizeof(self.matrix) + sys.getsizeof(eqlow.matrix)
        # est *= 1.2
        # avl = psutil.virtual_memory().available 
        # dec  =  max(int(math.ceil(est/avl)),1)
        # dec = 256
        # if X < dec:
            # ch = X
        # else:
            # ch = int(X / dec)
            
        ch = int(size / 16)

        for i in range(0, size - (size % ch), ch):
            c[range(i, i+ch), :] = np.dot(a[range(i, i+ch), :], b)

        left = size - (size % ch)
        c[range(left, size), :] = np.dot(a[range(left, size), :], b)

        return c

    def transpose(self, node: Node):
        matrix = node.transpose(self.matrix)
        const = Equation._get_const(
            node,
            torch.ones(node.output_size, dtype=torch.bool, device=self.config.DEVICE)
        )
        const = (self.matrix @ const) + self.const
        
        return Equation(matrix, const, self.config)


    def interval_transpose(self, node, bound):
        assert node.has_relu_activation() is True, "Interval transpose is not supported for nodes without relu activation."

        lower_slope = node.to_node[0].get_lower_relaxation_slope()
        lower_const = Equation._get_const(node, np.ones(node.output_size, bool))
        upper_const = lower_const.detach().clone()
        upper_slope = node.to_node[0].get_upper_relaxation_slope()
        lower_const *= lower_slope
        upper_const *= upper_slope
        upper_const[node.to_node[0].get_unstable_flag()]  -= \
            upper_slope[node.to_node[0].get_unstable_flag()] * \
            node.bounds.lower.flatten()[node.to_node[0].get_unstable_flag()]

        if bound == 'lower':
            plus = self._get_plus_matrix(force_float=True) * lower_slope
            minus = self._get_minus_matrix(force_float=True) * upper_slope
            # delegate to numpy sound calculation
            if self.config.interval_in_bs is True and self.config.gpu_in_bs is False:
                lower_const_intv = float_array2np_interval_cpu_array(lower_const.reshape(lower_const.size()[0], -1))
                upper_const_intv = float_array2np_interval_cpu_array(upper_const.reshape(upper_const.size()[0], -1))
                const = get_lower(
                    add(mat_mul(self._get_plus_matrix(),lower_const_intv,True),
                        mat_mul(self._get_minus_matrix(),upper_const_intv,True),
                        True)
                ).reshape(-1)
            # delefate to gemmc API sound calculation
            elif self.config.interval_in_bs is True and self.config.gpu_in_bs is True:
                lower_const_intv = torch_float_array2pinterval_gpu_array(lower_const.reshape(lower_const.size()[0], -1))
                upper_const_intv = torch_float_array2pinterval_gpu_array(upper_const.reshape(upper_const.size()[0], -1))
                const = get_lower(
                    add(mat_mul(self._get_plus_matrix(),lower_const_intv,True),
                        mat_mul(self._get_minus_matrix(),upper_const_intv,True),
                        True)
                ).reshape(-1)
            # non-interval cases can be dealt with by torch
            else:
                const = self._get_plus_matrix() @ lower_const + \
                    self._get_minus_matrix() @ upper_const


        elif bound == 'upper':
            plus = self._get_plus_matrix(force_float=True)  * upper_slope
            minus = self._get_minus_matrix(force_float=True) * lower_slope
            # delegate to numpy sound calculation
            if self.config.interval_in_bs is True and self.config.gpu_in_bs is False:
                lower_const_intv = float_array2np_interval_cpu_array(lower_const.reshape(lower_const.size()[0], -1))
                upper_const_intv = float_array2np_interval_cpu_array(upper_const.reshape(upper_const.size()[0], -1))
                const = get_upper(
                    add(mat_mul(self._get_plus_matrix(),upper_const_intv,True),
                        mat_mul(self._get_minus_matrix(),lower_const_intv,True),
                        True)
                ).reshape(-1)
            # delefate to gemmc API sound calculation
            elif self.config.interval_in_bs is True and self.config.gpu_in_bs is True:
                lower_const_intv = torch_float_array2pinterval_gpu_array(lower_const.reshape(lower_const.size()[0], -1))
                upper_const_intv = torch_float_array2pinterval_gpu_array(upper_const.reshape(upper_const.size()[0], -1))
                const = get_upper(
                    add(mat_mul(self._get_plus_matrix(),upper_const_intv,True),
                        mat_mul(self._get_minus_matrix(),lower_const_intv,True),
                        True)
                ).reshape(-1)
            # non-interval cases can be dealt with by torch
            else:
                const = self._get_plus_matrix() @ upper_const + \
                self._get_minus_matrix() @ lower_const

        else:
            raise ValueError(f'Bound {bound} is not recognised.')

        matrix = node.transpose(plus) + node.transpose(minus)

        const += self.const

        return Equation(matrix, const, self.config)


    @staticmethod
    def derive(node: Node, flag: torch.Tensor, config: Config) -> Equation:

        if isinstance(node.from_node[0], Relu) and \
        node.from_node[0].get_unstable_count() == 0  and \
        node.from_node[0].get_active_count() == 0:      
            return Equation(
                np.zeros((torch.sum(flag), 0)), Equation._get_const(node, flag), config
            )

        return Equation(
            Equation._get_matrix(node, flag), Equation._get_const(node, flag), config
        )
        

    @staticmethod
    def _get_matrix(node: Node, flag: torch.Tensor) -> torch.Tensor:
        if isinstance(node, Conv):
            flag_size = torch.sum(flag).item()
            prop_flag = torch.zeros(node.get_input_padded_size(), dtype=torch.bool)
            
            prop_flag[node.get_non_pad_idxs()] = True
            pad = torch.ones(
                node.get_input_padded_size(),
                dtype=torch.long
            ) * node.input_size
            pad[prop_flag] = torch.arange(node.input_size)
            im2col = Conv.im2col(
                pad.reshape(node.get_input_padded_shape()),
                (node.krn_height, node.krn_width),
                node.strides
            )
            conv_indices = im2col.repeat(1, node.out_ch)[:, flag]
            conv_weights = node.kernels.permute(1, 2, 3, 0).reshape(-1, node.out_ch)
            conv_weights = torch.repeat_interleave(conv_weights, node.out_ch_sz, dim=1)[:, flag]
                                                # )repeat(
                # 1, node.out_ch_sz)[:, flag]
            matrix = torch.zeros((flag_size, node.input_size + 1), dtype=node.config.PRECISION)
            matrix[torch.arange(flag_size), conv_indices] = conv_weights

            return matrix[:, :node.input_size]

        elif type(node) in [Gemm, MatMul]:
            return node.weights[flag, :]

        elif isinstance(node, Add):
            if node.const is not None:
                return np.identity(node.input_size, dtype=node.config.PRECISION)[:, flag] 
            else:
                matrix = np.zeros((node.output_size, 2 * node.output_size), dtype=node.config.PRECISION)
                matrix[range(node.output_size), range(node.output_size)] = 1
                matrix[range(node.output_size), range(node.output_size, 2 * node.output_size)] = 1
        else:
            raise  TypeError(f'Node {node} is not supported')

    @staticmethod
    def _get_const(node: Node, flag: torch.Tensor) -> torch.Tensor:
        if isinstance(node, Conv):
            out_ch_size = int(node.output_size / node.out_ch)
            
            return torch.tile(node.bias, (out_ch_size, 1)).T.flatten()[flag]

        elif isinstance(node, Gemm):
            return node.bias[flag]

        elif isinstance(node, MatMul):
            return np.zeros(node.output_size, dtype=node.weights.dtype)

        elif type(node) in [Sub, Add]:
            return torch.zeros(
                node.output_size, dtype=node.const.dtype, device=node.const.device
            )

        else:
            raise  TypeError(f'Node {node} is not supported')


    # def set_lower_slope(self, lbound, ubound):
        # """ 
        # Sets the lower slopes for the equations, one for computing the lower
        # bounds during back-substitution and one for computing the upper bound.

        # Arguments:
            
            # lbound: vector of the lower slope for the lower bound.
            
            # ubound: vector of the lower slope for the upper bound.

        # Returns:

            # None
        # """
        # self.lower_slope_l_bound = lbound
        # self.lower_slope_u_bound = ubound
        # self.is_slope_optimised = True

    # def get_lower_slope(self, l, u, approx=ReluApproximation.MIN_AREA):
        # """
        # Derives the slope of the lower linear relaxation of the equations.

        # Arguments:

            # lower: vector of lower bounds of the equations
            
            # upper: vector of upper bounds of the equations

            # approx: ReluApproximation

        # Returns:

            # vector of the slopes.
        # """

        # slope = np.zeros(self.size)
        # l, u = l.flatten(), u.flatten()

        # for i in range(self.size):
            # if  l[i] >= 0:
                # slope[i] = 1
            # elif u[i] <= 0: 
                # pass
            # else:
                # if approx == ReluApproximation.ZERO:
                    # pass
                # elif approx == ReluApproximation.IDENTITY:
                    # slope[i] = 1
                # elif approx == ReluApproximation.PARALLEL:
                    # slope[i] = u[i] / (u[i] - l[i])
                # elif approx == ReluApproximation.MIN_AREA:
                    # if abs(l[i]) < u[i]: slope[i] = 1
                # elif approx == ReluApproximation.VENUS_HEURISTIC:
                    # if abs(l[i]) < u[i]: slope[i] = u[i] / (u[i] - l[i])
                # else:
                    # pass

        # return slope 
