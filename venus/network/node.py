# ************
# File: node.py
# Top contributors (to current version): 
# 	Panagiotis Kouvaros (panagiotis.kouvaros@gmail.com)
# This file is part of the Venus project.
# Copyright: 2019-2021 by the authors listed in the AUTHORS file in the
# top-level directory.
# License: BSD 2-Clause (see the file LICENSE in the top-level directory).
# Description: Node classes.
# ************


import itertools 
import math
import numpy as np
import torch

from venus.bounds.bounds import Bounds
from venus.common.configuration import Config
from venus.common.utils import ReluState

torch.set_num_threads(1)

class Node:
    
    id_iter = itertools.count()

    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        output_shape: tuple,
        config: Config,
        depth=None,
        bounds=Bounds(),
        id=None,
    ):
        """
        Arguments:

            from_node:
                list of input nodes.
            to_node:
                list of output nodes.
            input_shape:
                shape of the input tensor to the node.
            output_shape: 
                shape of the output tensor to the node.
            config:
                configuration.
            depth:
                the depth of the node.
            bounds:
                concrete bounds for the node.
        """
        self.id = next(Node.id_iter) if id is None else id
        self.from_node = from_node
        self.to_node = to_node
        self.input_shape = input_shape
        self.input_size = np.prod(input_shape)
        self.output_shape = output_shape
        self.output_size = np.prod(output_shape)
        self.depth = depth
        self.config = config
        self.out_vars =  np.empty(0)
        self.delta_vars = np.empty(0)
        self.output = None
        self.outputs = None
        self.bounds = bounds

    def get_outputs(self):
        """
        Constructs a list of the indices of the nodes in the layer.

        Returns: 

            list of indices.
        """
        if self.outputs is not None:
            outputs = self.outputs
        else:
            if len(self.output_shape) > 1:
                outputs =  [i for i in itertools.product(*[range(j) for j in self.output_shape])]
            else:
                outputs = list(range(self.output_size))

        return outputs

    def clean_vars(self):
        """
        Nulls out all MILP variables associate with the network.
        """
        self.out_vars = torch.empty(0)
        self.delta_vars = torch.empty(0)


    def has_relu_activation(self) -> bool:
        """
        Returns:

            Whether the output of the node is fed to a relu node.
        """
        for i in self.to_node:
            if isinstance(i, Relu):
                return True

        return False

    def add_from_node(self, node):
        """
        Adds an input node.

        Arguments:
            
            node:
                The input node.
        """
        if self.has_from_node(node.id) is not True:
            self.from_node.append(node)

    def add_to_node(self, node):
        """
        Adds an output node.

        Arguments:
            
            node:
                The output node.
        """
        if self.has_to_node(node.id) is not True:
            self.to_node.append(node)


    def has_to_node(self, id: int) -> bool:
        """
        Checks whether the node points to another.

        Arguments:
            
            id:
                The id of the node with which connection to check.

        Returns:
            
            Whether the current node points to node with id "id".
        """
        for i in self.to_node:
            if i.id == id:
                return True

        return False

    def has_from_node(self, id: int) -> bool:
        """
        Checks whether the node is pointed to by another.

        Arguments:
            
            id:
                The id of the node with which connection to check.

        Returns:
            
            Whether the node with id "id" points to the current node.
        """
        for i in self.from_node:
            if i.id == id:
                return True

        return False

    def is_head_node(self) -> bool:
        """
        Returns:
            Whether the node is the first in the network.
        """
        return isinstance(self.from_node[0], Input)

    def is_tail_node(self) -> bool:
        """
        Returns:
            Whether the node is the first in the network.
        """
        return len(self.to_node) == 0


class Constant(Node):
    def __init__(self, to_node: list, const: torch.tensor, config: Config, id: int=None):
        """
        Argumnets:
            
            const:
                Constant matrix.
            to_node:
                list of output nodes.
            depth:
                the depth of the node.
        """
        super().__init__(
            [],
            to_node,
            const.shape,
            const.shape,
            config,
            depth=0,
            bounds=Bounds(config, const, const),
            id=id
        )

    def copy(self):
        """
        Copies the node.
        """
        return Constant(
            self.bounds.lower.detach().clone(),
            self.to_node,
            self.config,
            id=self.id
        )

class Input(Node):
    def __init__(self, bounds:torch.tensor, config: Config, id: int=None):
        """
        Argumnets:
            
            lower:
                array of lower bounds of input nodes.
            upper:
                array of upper bounds of input nodes.
        """
        super().__init__(
            [],
            [],
            bounds.lower.shape,
            bounds.lower.shape,
            config,
            depth=0,
            bounds=bounds,
            id=id
        )

    def copy(self):
        """
        Copies the node. 
        """
        return Input(
            self.bounds.copy(), self.config, id=self.id
        )

    def get_milp_var_size(self):
        """
        Returns the number of milp variables required for the milp encoding of
        the node.
        """
        return self.output_size

class Gemm(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        output_shape: tuple,
        weights: torch.tensor,
        bias: torch.tensor,
        config: Config,
        depth=None,
        bounds=Bounds(),
        id=None
    ):
        """
        Arguments:

            from_node:
                list of input nodes.
            to_node:
                list of output nodes.
            input_shape:
                shape of the input tensor to the node. 
            output_shape: 
                shape of the output tensor to the node.
            weights:
                weight matrix.
            bias:
                bias vector.
            config:
                configuration.
            depth:
                the depth of the node.
            bounds:
                concrete bounds for the node.
        """
        super().__init__(
            from_node,
            to_node,
            input_shape,
            output_shape,
            config,
            depth=depth,
            bounds=bounds,
            id=id
        )
        self.weights = weights
        self.bias = bias

    def copy(self):
        """
        Copies the node.
        """
        return Gemm(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.output_shape,
            self.weights,
            self.bias,
            self.config,
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )

    def get_milp_var_size(self):
        """
        Returns the number of milp variables required for the milp encoding of
        the node.
        """
        return self.output_size

    def get_bias(self, index: int) -> float:
        """
        Returns the bias of the given output.

        Arguments:

            index: 
                the index of the output.

        Returns:
            
            the bias.
        """
        return self.bias[index].item()

    def edge_weight(self, index1: int, index2: int) -> float:
        """
        Returns the weight of the edge between output with index1 of the
        current node and output with index2 of the previous node.

        Arguments:

            index1:
                index of the output of the current node.
            index2:
                index of the output of the previous node.

        Returns:
         
            the weight.
        """
        return self.weights[index1, index2].item()
    

    def forward(
        self,
        inp: torch.tensor=None,
        clip: str=None,
        add_bias: bool=True,
        save_output=False
    ) -> torch.tensor:
        """
        Computes the output of the node given an input.

        Arguments:
            inp:
                the input.
            clip:
                clips the weights to positive values if set to '+' and to
                negatives if set to '-'
            add_bias:
                whether to add bias
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp

        if clip is None:
            weights = self.weights

        elif clip == '+':
            weights = torch.clamp(self.weights, 0, math.inf)

        elif clip == '-':
            weights = torch.clamp(self.weights, -math.inf, 0)

        else:
            raise ValueError(f'Kernel clip value {clip} not recognised')
      
        output = weights @ inp

        if add_bias is True:
            output += self.bias

        if save_output:
            self.output = output

        return output

    def forward_numpy(self, inp: np.array=None, save_output=False) -> np.array:
        """
        Computes the output of the node given a numpy input.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp

        output = self.weights.numpy() @ inp + self.bias.numpy()

        if save_output is True:
            self.output = output

        return output


    def transpose(self, inp: torch.tensor) -> torch.tensor:
        """
        Computes the input to the node given an output.

        Arguments:
            inp:
                the output.
        Returns:
            the input of the node.
        """
        return inp @ self.weights


class MatMul(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        output_shape: tuple,
        weights: torch.tensor,
        config: Config,
        depth=None,
        bounds=Bounds(),
        id=None
    ):
        """
        Arguments:

            from_node:
                list of input nodes.
            to_node:
                list of output nodes.
            input_shape:
                shape of the input tensor to the node. 
            output_shape: 
                shape of the output tensor to the node.
            weights:
                weight matrix.
            config:
                configuration.
            depth:
                the depth of the node.
            bounds:
                concrete bounds for the node.
        """
        super().__init__(
            from_node,
            to_node,
            input_shape,
            output_shape,
            config,
            depth=depth,
            bounds=bounds,
            id=id
        )
        self.weights = weights

    def copy(self):
        """
        Copies the node.
        """
        return MatMul(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.output_shape,
            self.weights,
            self.config,
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )

    def numpy(self):
        """
        Copies the node with with numpy data.
        """
        return MatMul(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.output_shape,
            self.weights.numpy(),
            self.config,
            depth=self.depth,
            bounds=self.bounds,
            id=self.id
        )

    def get_milp_var_size(self):
        """
        Returns the number of milp variables required for the milp encoding of
        the node.
        """
        return self.output_size

    def edge_weight(self, index1: int, index2: int) -> float:
        """
        Returns the weight of the edge between output with index1 of the
        current node and output with index2 of the previous node.

        Arguments:

            index1:
                index of the output of the current node.
            index2:
                index of the output of the previous node.

        Returns:
         
            the weight.
        """
        return self.weights[index1, index2]
    

    def forward(self, inp: torch.tensor=None, clip: str=None, save_output=False) -> torch.tensor:
        """
        Computes the output of the node given an input.

        Arguments:
            inp:
                the input.
            clip:
                clips the weights to positive values if set to '+' and to
                negatives if set to '-'
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp

        if clip is None:
            weights = self.weights

        elif clip == '+':
            weights = tf.clip_by_value(self.weights, 0, math.inf)

        elif clip == '-':
            weights = tf.clip_by_value(self.weights, -math.inf, 0)

        else:
            raise ValueError(f'Kernel clip value {clip} not recognised')


        output = weights @ inp

        if save_output:
            self.output = output

        return output

    def forward_numpy(self, inp: np.array=None, save_output=False) -> np.array:
        """
        Computes the output of the node given a numpy input.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        output = self.weights.numpy() @ inp 

        if save_output is True:
            self.output = output

        return output

    def transpose(self, inp: torch.tensor) -> torch.tensor:
        """
        Computes the input to the node given an output.

        Arguments:
            inp:
                the output.
        Returns:
            the input of the node.
        """
        return inp @ self.weights



class Conv(Node):
    def __init__(
        self,
        from_node: list, 
        to_node: list,
        input_shape: tuple,
        output_shape: tuple,
        kernels: torch.tensor,
        bias: torch.tensor,
        padding: tuple,
        strides: tuple,
        config: Config,
        depth=None,
        bounds=Bounds(),
        id=None
    ):
        """
        Arguments:

            from_node:
                list of input nodes.
            to_node:
                list of output nodes.
            input_shape:
                shape of the input tensor to the node. 
            output_shape:
                shape of the output tensor to the node.
            kernels:
                the kernels.
            bias:
                bias vector.
            padding:
                the padding.
            strides:
                the strides.
            config:
                configuration.
            depth:
                the depth of the node.
            bounds:
                the concrete bounds of the node.
        """
        super().__init__(
            from_node,
            to_node,
            input_shape,
            output_shape,
            config,
            depth=depth,
            bounds=bounds,
            id=id
        )
        assert len(input_shape) in [3, 4]

        self.kernels = kernels
        self.bias = bias
        self.padding = padding
        self.strides = strides
        if len(input_shape) == 3:
            _, self.in_height, self.in_width = input_shape
            _, self.out_height, self.out_width = output_shape
        else:
            _, _, self.in_height, self.in_width = input_shape
            _, _, self.out_height, self.out_width = output_shape
        self.out_ch,  self.in_ch, self.krn_height, self.krn_width = kernels.shape
        self.out_ch_sz = int(self.output_size / self.out_ch)

    def copy(self):
        """
        Copies the node.
        """
        return Conv(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.output_shape,
            self.kernels,
            self.bias,
            self.padding,
            self.strides,
            self.config,
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )

    def numpy(self):
        """
        Copies the node with numpy data.
        """
        return Conv(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.output_shape,
            self.kernels.numpy(),
            self.bias.numpy(),
            self.padding,
            self.strides,
            self.config,
            depth=self.depth,
            bounds=self.bounds,
            id=self.id
        )
         
    def get_milp_var_size(self):
        """
        Returns the number of milp variables required for the milp encoding of
        the node.
        """
        return self.output_size

    def get_bias(self, index: tuple):
        """
        Returns the bias of the given output.

        Arguments:

            index: 
                the index of the output.

        Returns:
            
            the bias.
        """
        return self.bias[index[-1]]

    def edge_weight(self, index1: tuple, index2: tuple):
        """
        Returns the weight of the edge between output with index1 of the
        current node and output with index2 of the previous node.

        Arguments:

            index1:
                index of the output of the current node.
            index2:
                index of the output of the previous node.

        Returns:
         
            the weight.
        """
        height_start = index1[0] * self.strides[0] - self.padding[0]
        height = index2[0] - height_start
        width_start = index1[1] * self.strides[1] - self.padding[1]
        width = index2[1] - width_start

        return self.kernels[index1[0]][index2[0]][height][width]
        return self.kernels[index1[0]][index2[0]][height][width]


    def forward(
        self,
        inp: np.array=None,
        clip=None,
        add_bias=True,
        save_output=False
    ) -> torch.tensor:
        """
        Computes the output of the node given an input.

        Arguments:
            inp:
                the input.
            clip:
                clips the weights to positive values if set to '+' and to
                negatives if set to '-'
            add_bias:
                whether to add bias
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp

        if clip is None:
            kernels = self.kernels

        elif clip == '+':
            kernels = torch.clamp(self.kernels, 0, math.inf)

        elif clip == '-':
            kernels = torch.clamp(self.kernels, -math.inf, 0)

        else:
            raise ValueError(f'Kernel clip value {kernel_clip} not recognised')

        output = torch.nn.functional.conv2d(
            inp,
            kernels,
            stride=self.strides,
            padding=self.padding,
        ).flatten()

        if add_bias is True:
            output = output.flatten() + torch.tile(self.bias, (self.out_ch_sz, 1)).T.flatten()

        output = output.reshape(self.output_shape)
                    
        if save_output:
            self.output = output

        return output


    def forward_numpy(self, inp: np.array=None, save_output=False) -> np.array:
        """
        Computes the output of the node given an input.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp

        padded_inp = Conv.pad(inp, self.padding)

        inp_strech = Conv.im2col(
            padded_inp, (self.krn_height, self.krn_width), self.strides
        )

        kernel_strech = self.kernels.reshape(self.out_ch, -1).numpy()

        output = kernel_strech @ inp_strech
        output = output.flatten() + np.tile(self.bias.numpy(), (self.out_ch_sz, 1)).T.flatten()
        output = output.reshape(self.output_shape)

        if save_output is True:
            self.output = output

        return output

    def transpose(self, inp: torch.tensor) -> torch.tensor:
        """
        Computes the input to the node given an output.

        Arguments:

            inp:
                the output.
                
        Returns:
            
            the input of the node.
        """
        out_pad_height = self.in_height - (self.out_height - 1) * self.strides[0] + 2 * self.padding[0] - self.krn_height
        out_pad_width = self.in_width - (self.out_width - 1) * self.strides[0] + 2 * self.padding[0] - self.krn_width

        return torch.nn.functional.conv_transpose2d(
            inp.reshape((inp.shape[0], self.out_ch, self.out_height, self.out_width)),
            self.kernels,
            stride=self.strides,
            padding=self.padding,
            output_padding=(out_pad_height, out_pad_width)
        ).reshape(inp.shape[0], - 1)

        # return tf.nn.conv2d_transpose(
            # inp.reshape((inp.shape[0],) + self.output_shape),
            # self.kernels.transpose(2, 3, 1, 0),
            # (inp.shape[0], ) + self.input_shape,
            # self.strides,
            # padding = [
                # [0, 0],
                # [self.padding[0], self.padding[0]],
                # [self.padding[1], self.padding[1]],
                # [0, 0]
            # ]
        # ).numpy().reshape(inp.shape[0], -1)


    def get_non_pad_idxs(self) -> torch.tensor:
        """
        Returns:

            Indices of the original input whithin the padded one.
        """
        if len(self.input_shape) == 3:
            in_ch, height, width = self.input_shape
        else:
            _, in_ch, height, width = self.input_shape

        pad_height, pad_width = self.padding
        size = self.get_input_padded_size()
        non_pad_idxs = torch.arange(size, dtype=torch.long).reshape(
            self.in_ch,
            height + 2 * pad_height,
            width + 2 * pad_width
        )[:, pad_height:height + pad_height, pad_width :width + pad_width].flatten()

        return non_pad_idxs

    def get_input_padded_size(self) -> int:
        """
        Returns:

                Size of the padded input.
        """
        return (self.in_height + 2 * self.padding[0]) * (self.in_width + 2 * self.padding[1]) * self.in_ch
        

    def get_input_padded_shape(self) -> tuple:
        """
        Computes the shape of padded input.
        """  
        return (
            self.in_ch,
            self.in_height + 2 * self.padding[0],
            self.in_width + 2 * self.padding[1],
        )
        

    @staticmethod
    def compute_output_shape(in_shape: tuple, weights_shape: tuple, padding: tuple, strides: tuple) -> tuple:
        """
        Computes the output shape of a convolutional layer.

        Arguments:
            in_shape:
                shape of the input tensor to the layer.
            weights_shape:
                shape of the kernels of the layer.
            padding:
                pair of int for the width and height of the padding.
            strides:
                pair of int for the width and height of the strides.
        Returns:
            tuple of the output shape
        """
        assert len(in_shape) in [3, 4]

        if len(in_shape) == 3:
            in_ch, in_height, in_width = in_shape
        else:
            batch, in_ch, in_height, in_width = in_shape

        out_ch, _, k_height, k_width = weights_shape

        out_height = int(math.floor(
            (in_height - k_height + 2 * padding[0]) / strides[0] + 1
        ))
        out_width = int(math.floor(
            (in_width - k_width + 2 * padding[1]) / strides[1] + 1
        ))
      
        if len(in_shape) == 3:
            return out_ch, out_height, out_width
        else:
            return batch, out_ch, out_height, out_width

    @staticmethod
    def pad(inp: torch.tensor, padding: tuple, values: tuple=(0,0)) -> torch.tensor:
        """
        Pads a given matrix with constants.

        Arguments:
            
            inp:
                matrix.
            padding:
                the padding.
            values:
                the constants.

        Returns

            padded inp.
        """
        assert(len(inp.shape)) in [3, 4]

        if padding == (0, 0):
            return inp
       
        if len(inp.shape) == 3:
            padding = ((0, 0), padding, padding)
        else:
            padding = ((0, 0), (0, 0), padding, padding)

        return np.pad(inp, padding, 'constant', constant_values=values)


    @staticmethod
    def im2col(matrix: torch.tensor, kernel_shape: tuple, strides: tuple, indices: bool=False) -> torch.tensor:
        """
        im2col function.

        Arguments:
            matrix:
                The matrix.
            kernel_shape:
                The kernel shape.
            strides:
                The strides of the convolution.
            indices:
                Whether to select indices (true) or values (false) of the matrix.
        Returns:
            im2col matrix
        """
        assert len(matrix.shape) in [3, 4], f"{len(matrix.shape)}-D is not supported."
        assert type(matrix) in [torch.Tensor, np.ndarray], f"{type(matrix)} matrices are not supported."

        opr = torch if isinstance(matrix, torch.Tensor) else np

        filters, height, width = matrix.shape[-3:]
        row_extent = height - kernel_shape[0] + 1
        col_extent = width - kernel_shape[1] + 1

        # starting block indices
        start_idx = opr.arange(kernel_shape[0])[:, None] * height + np.arange(kernel_shape[1])
        start_idx = start_idx.flatten()[None, :]
        offset_filter = np.arange(
            0, filters * height * width, height * width
        ).reshape(-1, 1)
        start_idx = start_idx + offset_filter


        # offsetted indices across the height and width of A
        offset_idx = opr.arange(row_extent)[:, None][::strides[0]] * height +  opr.arange(0, col_extent, strides[1])

        # actual indices
        if indices is True:
            return start_idx.ravel()[:, None] + offset_idx.ravel()

        return opr.take(matrix, start_idx.ravel()[:, None] + offset_idx.ravel())


class Relu(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        config: Config,
        depth=None,
        bounds=Bounds(),
        id=None
    ):
        """
        Arguments:

            from_node:
                list of input nodes.
            to_node:
                list of output nodes.
            input_shape:
                shape of the input tensor to the node. 
            config:
                configuration.
            depth:
                the depth of the node.
            bounds:
                concrete bounds for the node.
        """
        super().__init__(
            from_node,
            to_node,
            input_shape,
            input_shape,
            config,
            depth=depth,
            bounds=bounds,
            id=id
        )
        self.state = np.array(
            [ReluState.UNSTABLE] * self.output_size,
            dtype=ReluState,
        ).reshape(self.output_shape)
        self.dep_root = np.array(
            [False] * self.output_size,
            dtype=bool
        ).reshape(self.output_shape)
        self.active_flag = None
        self.inactive_flag = None
        self.stable_flag = None
        self.unstable_flag = None
        self.propagation_flag = None
        self.stable_count = None
        self.unstable_count = None
        self.active_count = None
        self.inactive_count = None
        self.propagation_count = None

    def copy(self):
        """
        Copies the node.
        """
        relu = Relu(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.config,
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )
        relu.state = self.state.copy()
        relu.dep_root = self.dep_root.copy()

        return relu

    def get_milp_var_size(self):
        """
        Returns the number of milp variables required for the milp encoding of
        the node.
        """
        return 2 * self.output_size

    def forward(self, inp: torch.tensor=None, save_output=None) -> torch.tensor:
        """
        Computes the output of the node given an input.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node.
        Returns:
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp

        output = torch.clamp(inp, 0, math.inf)

        if save_output:
            self.output = output

        return output

    def forward_numpy(self, inp: np.array=None, save_output=None) -> np.array:
        """
        Computes the output of the node given a numpy input.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node.
        Returns:
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp

        output = np.clip(inp, 0, math.inf)

        if save_output:
            self.output = output

        return output


    def reset_state_flags(self):
        """
        Resets calculation flags for relu states
        """
        self.active_flag = None
        self.inactive_flag = None
        self.stable_flag = None
        self.unstable_flag = None
        self.propagation_flag = None
        self.propagation_flag = None
        self.unstable_count = None
        self.active_count = None
        self.inactive_count = None
        self.propagation_count = None

    def is_active(self, index):
        """
        Detemines whether a given ReLU node is stricly active.

        Arguments:

            index: 
                the index of the node.

        Returns:

            bool expressing the active state of the given node.
        """
        cond1 = self.from_node[0].bounds.lower[index] >= 0
        cond2 = self.state[index] == ReluState.ACTIVE

        return cond1 or cond2

    def is_inactive(self, index: tuple):
        """
        Determines whether a given ReLU node is strictly inactive.

        Arguments:

            index:
                the index of the node.

        Returns:

            bool expressing the inactive state of the given node.
        """
        cond1 = self.from_node[0].bounds.upper[index] <= 0
        cond2 = self.state[index] == ReluState.INACTIVE

        return cond1 or cond2

    def is_stable(self, index: tuple, delta_val: float=None):
        """
        Determines whether a given ReLU node is stable.

        Arguments:

            index:
                the index of the node.

            delta_val:
                the value of the binary variable associated with the node. if
                set, the value is also used in conjunction with the node's
                bounds to determined its stability.

        Returns:

            bool expressing the stability of the given node.
        """
        cond0a = self.from_node[0].bounds.lower[index].item() >= 0
        cond0b = self.from_node[0].bounds.upper[index].item() <= 0
        cond1 = cond0a or cond0b
        cond2 = self.state[index] != ReluState.UNSTABLE
        cond3 = False if delta_val is None else delta_val in [0, 1]

        return cond1 or cond2 or cond3

    def get_active_flag(self) -> torch.tensor:
        """
        Returns an array of activity statuses for each ReLU node.
        """
        if self.active_flag is None:
            self.active_flag = self.from_node[0].bounds.lower.flatten() > 0

        return self.active_flag

    def get_active_count(self) -> int:
        """
        Returns the total number of active Relu nodes.
        """
        if self.active_count is None:
            self.active_count = torch.sum(self.get_active_flag())

        return self.active_count

    def get_inactive_flag(self) -> torch.tensor:
        """
        Returns an array of inactivity statuses for each ReLU node.
        """
        if self.inactive_flag is None:
            self.inactive_flag = self.from_node[0].bounds.upper.flatten() <= 0

        return self.inactive_flag

    def get_inactive_count(self) -> int:
        """
        Returns the total number of inactive ReLU nodes.
        """
        if self.inactive_count is None:
            self.inactive_count = torch.sum(self.get_inactive_flag())

        return self.inactive_count

    def get_unstable_flag(self) -> torch.tensor:
        """
        Returns an array of instability statuses for each ReLU node.
        """
        if self.unstable_flag is None:
            self.unstable_flag = torch.logical_and(
                self.from_node[0].bounds.lower < 0,
                self.from_node[0].bounds.upper > 0
            ).flatten()

        return self.unstable_flag

    def get_unstable_count(self) -> int:
        """
        Returns the total number of unstable nodes.
        """
        if self.unstable_count is None:
            self.unstable_count = torch.sum(self.get_unstable_flag())

        return self.unstable_count

    def get_stable_flag(self) -> torch.tensor:
        """
        Returns an array of instability statuses for each ReLU node.
        """
        if self.stable_flag is None:
            self.stable_flag = torch.logical_or(
                self.get_active_flag(),
                self.get_inactive_flag()
            ).flatten()

        return self.stable_flag

    def get_stable_count(self) -> int:
        """
        Returns the total number of unstable nodes.
        """
        if self.stable_count is None:
            self.stable_count = torch.sum(self.get_stable_flag()).item()

        return self.stable_count

    def get_propagation_flag(self) -> torch.tensor:
        """
        Returns an array of sip propagation statuses for each node.
        """
        if self.propagation_flag is None:
            self.propagation_flag = torch.logical_or(
                self.get_active_flag(),
                self.get_unstable_flag()
            ).flatten()

        return self.propagation_flag

    def get_propagation_count(self) -> int:
        """
        Returns the total number of sip propagation nodes.
        """
        if self.propagation_count is None:
            self.propagation_count = torch.sum(self.get_propagation_flag())

        return self.propagation_count
 
    def get_upper_relaxation_slope(self) -> torch.tensor:
        """
        Returns:

        The upper relaxation slope for each of the ReLU nodes.
        """
        slope = torch.zeros(
            self.output_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        )
        upper = self.from_node[0].bounds.upper.flatten()[self.get_unstable_flag()]
        lower = self.from_node[0].bounds.lower.flatten()[self.get_unstable_flag()]
        slope[self.get_unstable_flag()] = upper /  (upper - lower)
        slope[self.get_active_flag()] = 1.0
        
        return slope

    def get_lower_relaxation_slope(self):
        """
        Returns:

        The upper relaxation slope for each of the ReLU nodes.
        """
        slope = torch.ones(
            self.output_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        )
        upper = self.from_node[0].bounds.upper.flatten()
        lower = self.from_node[0].bounds.lower.flatten()
        idxs = abs(lower) >=  upper
        slope[idxs] = 0.0
        slope[self.get_inactive_flag()] = 0.0
        slope[self.get_active_flag()] = 1.0

        return slope


class Reshape(Node):
    def __init__(
        self,
        from_node: list, 
        to_node: list,
        input_shape: tuple,
        output_shape: tuple,
        config: Config,
        depth=None,
        bounds=Bounds(),
        id=None
    ):
        """
        Arguments:

            from_node:
                list of input nodes.
            to_node:
                list of output nodes.
            input_shape:
                shape of the input tensor to the node. 
            output_shape: 
                shape of the output tensor to the node.
            config:
                configuration.
            depth:
                the depth of the node.
            bounds:
                concrete bounds for the node.
        """
        super().__init__(
            from_node,
            to_node,
            input_shape,
            output_shape,
            config,
            depth=self.depth,
            bounds=self.bounds,
            id=id
        )

    def copy(self):
        """
        Copies the node.
        """
        return Reshape(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.output_shape,
            self.config,
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )

    def get_milp_var_size(self):
        """
        Returns the number of milp variables required for the milp encoding of
        the node.
        """
        return 0

class Flatten(Node):
    def __init__(
        self,
        from_node: list, 
        to_node: list,
        input_shape: tuple,
        config: Config,
        depth=None,
        bounds=Bounds(),
        id=None
    ):
        """
        Arguments:

            from_node:
                list of input nodes.
            to_node:
                list of output nodes.
            input_shape:
                shape of the input tensor to the node. 
            config:
                configuration.
            depth:
                the depth of the node.
            bounds:
                concrete bounds for the node.
        """
        super().__init__(
            from_node,
            to_node,
            input_shape,
            (np.prod(input_shape),),
            config,
            depth=depth,
            bounds=bounds,
            id=id
        )

    def copy(self):
        """
        Copies the node.
        """
        return Flatten(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.config,
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )

    def get_milp_var_size(self):
        """
        Returns the number of milp variables required for the milp encoding of
        the node.
        """
        return 0

    def forward(self, inp: torch.tensor=None, save_output=False) -> torch.tensor:
        """
        Computes the output of the node given an input.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node.
        Returns: 
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp

        output = inp.flatten()

        if save_output:
            self.output = output

        return output

    def forward_numpy(self, inp: np.ndarray=None, save_output=False) -> np.ndarray:
        """
        Computes the output of the node given a numpy input.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node.
        Returns: 
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp

        output = inp.flatten()

        if save_output:
            self.output = output

        return output


class Sub(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        config: Config,
        const=None,
        depth=None,
        bounds=Bounds(),
        id=None
    ):
        """
        Arguments:

            from_node:
                list of input nodes.
            to_node:
                list of output nodes.
            input_shape:
                shape of the input tensor to the node.
            config:
                configuration.
            const:
                Constant second term is set.
            depth:
                the depth of the node.
            bounds:
                concrete bounds for the node.
        """
        super().__init__(
            from_node,
            to_node,
            input_shape,
            input_shape,
            config,
            depth=depth,
            bounds=bounds,
            id=id
        )
        self.const = const

    def copy(self):
        """
        Copies the node.
        """
        return Sub(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.config,
            const=None if self.const is None else self.const.detach().clone(),
            depth=self.depth,
            bounds=self.bounds,
            id=self.id
        )

    def numpy(self):
        """
        Copies the node with numpy data.
        """
        return Sub(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.config,
            const=None if self.const is None else self.const.numpy(),
            depth=self.depth,
            bounds=self.bounds,
            id=self.id
        )

    def get_milp_var_size(self):
        """
        Returns the number of milp variables required for the milp encoding of
        the node.
        """
        return self.output_size


    def forward(self, inp1: torch.tensor=None, inp2: torch.tensor=None, save_output=False) -> torch.tensor:
        """
        Computes the output of the node given an input.

        Arguments:
            inp1:
                the first input.
            inp2:
                the second input. If not set then the const of the node is
                taken as second input.
            save_output:
                Whether to save the output in the node.
        Returns:
            the output of the node.
        """

        assert inp1 is not None or self.from_node[0].output is not None
        assert inp2 is not None or self.const is not None

        inp1 = self.from_node[0].output if inp1 is None else inp1
        inp2 = self.const if inp2 is None else inp2

        output = inp1 - inp2

        if save_output:
            self.output = output

        return output

    def forward_numpy(self, inp1: np.array, inp2: np.array=None) -> np.array:
        """
        Computes the output of the node given a numpy input.

        Arguments:
            inp1:
                the first input.
            inp2:
                the second input. If not set then the const of the node is
                taken as second input.
            save_output:
                Whether to save the output in the node.
        Returns:
            the output of the node.
        """

        assert inp2 is not None or self.const is not None

        inp2 = self.const.numpy() if inp2 is None else inp2

        output = inp1 - inp2

        return output

    def transpose(self, inp: torch.tensor) -> torch.tensor:
        """
        Computes the input to the node given an output.

        Arguments:

            inp:
                the output.
                
        Returns:
            
            the input of the node.
        """
        if self.const is not None:
            return inp - self.const.flatten()

        sub = np.zeros(
            self.output_size, 2 * self.output_size,
            dtype=self.config.PRECISION
        )

        return np.hstack([inp, - inp])

class Add(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        config: Config,
        const=None,
        depth=None,
        bounds=Bounds(),
        id=None
    ):
        """
        Arguments:

            from_node:
                list of input nodes.
            to_node:
                list of output nodes.
            input_shape:
                shape of the input tensor to the node.
            config:
                configuration.
            const:
                Constant second term is set.
            depth:
                the depth of the node.
            bounds:
                concrete bounds for the node.
        """
        super().__init__(
            from_node,
            to_node,
            input_shape,
            input_shape,
            config,
            depth=depth,
            bounds=bounds,
            id=id
        )
        self.const = const

    def copy(self):
        """
        Copies the node.
        """
        return Add(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.config,
            const = None if self.const is None else self.const.copy(),
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )

    def numpy(self):
        """
        Copies the node with numpy data.
        """
        return Add(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.config,
            const = None if self.const is None else self.const.numpy(),
            depth=self.depth,
            bounds=self.bounds,
            id=self.id
        )

    def get_milp_var_size(self):
        """
        Returns the number of milp variables required for the milp encoding of
        the node.
        """
        return self.output_size

    def forward(self, inp1: torch.tensor=None, inp2: torch.tensor=None, save_output=False) -> torch.tensor:
        """
        Computes the output of the node given an input.

        Arguments:
            inp1:
                the first input.
            inp2:
                the second input. If not set then the const of the node is
                taken as second input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        assert inp1 is not None or self.from_node[0].output is not None
        assert inp2 is not None or self.const is not None

        inp1 = self.from_node[0].output if inp is None else inp1
        inp2 = self.const if inp2 is None else inp2

        output = tf.add(inp1, inp2)

        if save_output:
            self.output = output

        return output

    def forward_numpy(self, inp1: np.array, inp2: np.array=None) -> np.array:
        """
        Computes the output of the node given a numpy input.

        Arguments:
            inp1:
                the first input.
            inp2:
                the second input. If not set then the const of the node is
                taken as second input.
            save_output:
                Whether to save the output in the node.
        Returns:
            the output of the node.
        """

        assert inp2 is not None or self.const is not None

        inp2 = self.const.numpy() if inp2 is None else inp2

        output = inp1 + inp2

        return output

    def transpose(self, inp: torch.tensor) -> torch.tensor:
        """
        Computes the input to the node given an output.

        Arguments:

            inp:
                the output.
                
        Returns:
            
            the input of the node.
        """
        if self.const is not None:
            return inp + self.const.flatten()

        return np.hstack([inp, inp])


class BatchNormalization(Node):
    def __init__(
        self,
        from_node: list,
        to_node: list,
        input_shape: tuple,
        scale: torch.tensor,
        bias: torch.tensor,
        input_mean: torch.tensor,
        input_var: torch.tensor,
        epsilon: float,
        config: Config,
        depth=None,
        bounds=Bounds(),
        id=None
    ):
        """
        Arguments:

            from_node:
                list of input nodes.
            to_node:
                list of output nodes.
            input_shape:
                shape of the input tensor to the node.
            scale:
                the scale tensor.
            bias:
                the bias tensor.
            input_mean:
                the mean tensor.
            input_var:
                the variance tensor.
            epsilon:
                the epsilon value.
            config:
                configuration.
            depth:
                the depth of the node.
            bounds:
                concrete bounds for the node.
        """
        super().__init__(
            from_node,
            to_node,
            input_shape,
            input_shape,
            config,
            depth=depth,
            bounds=bounds,
            id=id
        )
        self.scale = scale
        self.bias = bias
        self.input_mean = input_mean
        self.input_var = input_var
        self.epsilon = epsilon

    def copy(self):
        """
        Copies the node.
        """
        return BatchNormalization(
            self.from_node,
            self.to_node,
            self.input_shape,
            self.scale,
            self.bias,
            self.input_mean,
            self.input_var,
            self.epsilon,
            self.config,
            depth=self.depth,
            bounds=self.bounds.copy(),
            id=self.id
        )

    def get_milp_var_size(self):
        """
        Returns the number of milp variables required for the milp encoding of
        the node.
        """
        return self.output_size

    def forward(self, inp: torch.tensor=None, save_output=False) -> torch.tensor:
        """
        Computes the output of the node given an input.

        Arguments:
            inp:
                the input.
            save_output:
                Whether to save the output in the node. 
        Returns: 
            the output of the node.
        """
        assert inp is not None or self.from_node[0].output is not None
        inp = self.from_node[0].output if inp is None else inp1

        output = tf.nn.batch_normalization(
            self.input_mean,
            self.input_var,
            self.bias,
            self.scale,
            self.epsilon
        )
        # output = tf.subtract(inp, self.input_mean)
        # output = tf.divide(output, math.sqrt(self.input_var + self.epsilon))
        # output = tf.multiply(output, self.scale)
        # output = tf.add(output, self.bias)

        if save_output:
            self.output = output

        return output

    def transpose(self, inp: torch.tensor) -> torch.tensor:
        """
        Computes the input to the node given an output.

        Arguments:

            inp:
                the output.
                
        Returns:
            
            the input of the node.
        """
        return

