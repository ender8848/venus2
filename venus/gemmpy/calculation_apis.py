import numpy as np
from venus.gemmpy.Interval import Interval
import ctypes
import torch
import sys
import os
# use np for cpu calculation
# use torch for gpu calculation

# set dynamic linking llib here
file_path = os.path.dirname(os.path.abspath(__file__))
gemmlib = ctypes.CDLL(file_path + '/gemmGPU.so')
mmalib = ctypes.CDLL(file_path + '/mma.so')
# better put argtype setting here


def float_array2np_interval_cpu_array(arr:np.ndarray):
    """
    converts a real-valued numpy array to an interval numpy array
    args: 
        arr: array_like
    returns:
        an numpy array converted from arr
    """
    result = np.empty((arr.shape),dtype=Interval)
    
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            result[i,j] = Interval(arr[i,j], arr[i,j])
    return result


def torch_float_array2pinterval_gpu_array(arr:torch.Tensor):
    """
    converts a real-valued torch array to a pseudo interval torch array which uses continuous memory view to mimic an Interval array
    args: 
        arr: torch tensor
    returns:
        an pseudo interval torch array converted from arr
    """
    if not isinstance(arr, torch.Tensor):
        raise TypeError("arr must be torch tensor")
    result = torch.repeat_interleave(arr, 2, dim = 1)
    # convert to gpu memory
    result = result.to(torch.device('cuda'))
    # result = torch.zeros((arr.shape[0], arr.shape[1]*2), dtype = torch.float32, device=torch.device('cuda'))

    # for i in range(arr.shape[0]):
    #     for j in range(arr.shape[1]):
    #         result[i,2*j] = arr[i][j]
    #         result[i,2*j+1] = arr[i][j]
    return result


def get_upper(arr, ret = 'torch'):
    """
    converts an interval numpy array or torch pseudo interval tensor
    to corresponding upper-value float array
    args: 
        arr: numpy Interval array or torch pseudo interval array
        ret: return type, 'numpy' or 'torch'
    returns:
        upper value float array, return type is cpu array
    """
    if isinstance(arr, np.ndarray) and arr.dtype != np.float32: # have to use != coz np does not recognize Interval type
        if ret == 'numpy' or ret == 'np':
            result = np.empty((arr.shape), dtype = np.float32)
        elif ret == 'torch' or ret == 't':
            result = torch.empty((arr.shape), dtype = torch.float32)
        else:
            raise ValueError("ret must be 'numpy' or 'torch'")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                result[i,j] = arr[i,j].upper
        return result
    elif isinstance(arr, torch.Tensor):
        if ret == 'numpy' or ret == 'np':
            result = np.empty((arr.shape[0], arr.shape[1]//2), dtype = np.float32)
        elif ret == 'torch' or ret == 't':
            result = torch.empty((arr.shape[0], arr.shape[1]//2), dtype = torch.float32)
        else:
            raise ValueError("ret must be 'numpy' or 'torch'")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]//2):
                result[i,j] = arr[i,2*j+1]
        return result
    else:
        raise TypeError("arr must be numpy array or torch tensor")


def get_lower(arr, ret = 'torch'):
    """
    converts an interval numpy array or torch pseudo interval tensor
    to corresponding lower-value float array
    args: 
        arr: numpy Interval array or torch pseudo interval array
        ret: return type, 'numpy' or 'torch'
    returns:
        lower value float array, return type is cpu array
    """
    if isinstance(arr, np.ndarray) and arr.dtype != np.float32: # have to use != coz np does not recognize Interval type
        if ret == 'numpy' or ret == 'np':
            result = np.empty((arr.shape), dtype = np.float32)
        elif ret == 'torch' or ret == 't':
            result = torch.empty((arr.shape), dtype = torch.float32)
        else:
            raise ValueError("ret must be 'numpy' or 'torch'")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                result[i,j] = arr[i,j].lower
        return result
    elif isinstance(arr, torch.Tensor):
        if ret == 'numpy' or ret == 'np':
            result = np.empty((arr.shape[0], arr.shape[1]//2), dtype = np.float32)
        elif ret == 'torch' or ret == 't':
            result = torch.empty((arr.shape[0], arr.shape[1]//2), dtype = torch.float32)
        else:
            raise ValueError("ret must be 'numpy' or 'torch'")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]//2):
                result[i,j] = arr[i,2*j]
        return result
    else:
        raise TypeError("arr must be numpy array or torch tensor")


def mmaGPUPy(dest_dev: torch.Tensor, A_dev:torch.Tensor, B_dev:torch.Tensor, M, N):
    """
    perform matrix addition using C lib, works only for torch pseudo interval tensor 
    args:
        dest_dev: output tensor
        A_dev: input tensor 1
        B_dev: input tensor 2
        M: number of rows in A and B
        N: number of columns in A and B
    """
    M_c = ctypes.c_int(M)
    N_c = ctypes.c_int(N)
    dtype = ctypes.c_int(2) # namely interval float
    a_p = ctypes.cast(A_dev.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    b_p = ctypes.cast(B_dev.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    dest_p = ctypes.cast(dest_dev.data_ptr(), ctypes.POINTER(ctypes.c_void_p))
    mmalib.mma2DGPUPy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    mmalib.mma2DGPUPy(dest_p, a_p, b_p, M_c, N_c, dtype)


def add(A, B, interval:bool):
    """
    perform matrix addition on numpy float array, interval array, torch float array or torch pseudo interval array
    args:
        A: input array 1
        B: input array 2
    returns:
        new array A + B, same type as A and B
    """
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor) and interval:
        dest = torch.empty_like(A)
        mmaGPUPy(dest, A, B, A.shape[0], A.shape[1])
        return dest
    # all other cases can be dealt with by + operator
    return A + B


def gemmGPUPy(dest_dev:torch.Tensor, A_dev:torch.Tensor, B_dev:torch.Tensor, M, N, K, bias_dev:torch.Tensor = None):
    """
    perform matrix multiplication using c lib, works only for torch pseudo tensor and requires data initialized in GPU memory
    args:
        dest_dev: output tensor
        A_dev: lhs tensor
        B_dev: rhs tensor
        M: number of rows in A and dest
        N: number of columns in B and dest
        K: number of columns in A and rows in B
        bias_dev: optional bias tensor
    """
    M_c = ctypes.c_int(M)
    N_c = ctypes.c_int(N)
    K_c = ctypes.c_int(K)
    dtype = ctypes.c_int(2) # namely interval float
    a_p = ctypes.cast(A_dev.data_ptr(), ctypes.c_void_p)
    b_p = ctypes.cast(B_dev.data_ptr(), ctypes.c_void_p)
    is_host_c = ctypes.c_bool(False)
    dest_p = ctypes.cast(dest_dev.data_ptr(), ctypes.c_void_p)
    if bias_dev is None:
        # create a nullptr
        bias_c = ctypes.c_void_p(0)
    else:
        bias_c = ctypes.cast(bias_dev.data_ptr(), ctypes.c_void_p)
    gemmlib.gemmGPUPy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_bool]
    gemmlib.gemmGPUPy(a_p, b_p, dest_p, M_c, N_c, K_c, dtype, bias_c, is_host_c)


def mat_mul(A, B, interval:bool):
    """
    perform matrix multiplication A @ B
    args:
        A: lhs of matrix multiplication, can be numpy float array, numpy interval array, torch float array or torch pseudo interval array
        B: rhs of matrix multiplication, can be numpy float array, numpy interval array, torch float array or torch pseudo interval array
        interval: whether to use sound calculation
    returns:
        new np array A @ B, same type as A and B
    """
    if isinstance(A, np.ndarray):
        return A @ B
    if not interval:
        return A @ B
    # GPU and interval case
    dest = torch.zeros((A.shape[0], B.shape[1]), dtype = torch.float32, device='cuda') # try replace zero with empty
    gemmGPUPy(dest, A, B, A.shape[0], B.shape[1]//2, B.shape[0])
    return dest
    

def gemm(A, B, bias, interval:bool):
    """
    performs gemm dest = A @ B + bias
    args:
        A: lhs of matrix multiplication, can be numpy float array, numpy interval array, torch float array or torch pseudo interval array
        B: rhs of matrix multiplication, can be numpy float array, numpy interval array, torch float array or torch pseudo interval array
        bias: bias to be added, can be numpy float array, numpy interval array, torch float array or torch pseudo interval array
        interval: whether to use sound calculation
    returns:
        dest of matrix multiplication, same type as A and B
    """

    if bias is None:
        return mat_mul(A, B, interval)
    if isinstance(A, np.ndarray):
        return A @ B + bias
    if not interval:
        return A @ B + bias
    # interval case
    dest = torch.zeros((A.shape[0], B.shape[1]), dtype=torch.float32, device='cuda')
    gemmGPUPy(dest, A, B, A.shape[0], B.shape[1]//2, B.shape[0],bias)
    return dest