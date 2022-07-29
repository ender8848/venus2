import numpy as np
import torch
from calculation_apis import *
from Interval import print_2d_array
from math import nextafter, inf
import time
import csv

def can_convert_real_numbered_np_array_to_Interval_np_array():
    """
    test function np_array_float2interval
    """
    a = np.array([[1,2],[3,4]], dtype = np.float32)
    b = float_array2np_interval_cpu_array(a)
    assert(b.dtype == object)
    assert(b.shape == a.shape)
    assert(b[0,0] == Interval(1,1))
    assert(b[0,1] == Interval(2,2))
    assert(b[1,0] == Interval(3,3))
    assert(b[1,1] == Interval(4,4))


def can_convert_float_torch_array_to_pesudo_interval_array():
    """
    test function torch_array_float2pinterval
    """
    # gpu case
    a = torch.tensor([[1,2],[3,4]], dtype = torch.float32, device='cuda')
    b = torch_float_array2pinterval_gpu_array(a)
    assert(b.dtype == a.dtype)
    # check if b is gpu memory
    assert(b.device == torch.device('cuda') or b.device == torch.device('cuda:0'))
    assert(torch.equal(b, torch.tensor([[1,1,2,2],[3,3,4,4]], dtype = torch.float32, device = 'cuda')))


def can_convert_numpy_Interval_array_to_upper():
    """
    test function get_upper for numpy interval array
    """
    # numpy interval case
    a = np.array([[Interval(1,1), Interval(1,2)], [Interval(1,3), Interval(1,4)]])
    b = get_upper(a, ret = 'np')
    assert(b.dtype == np.float32)
    assert(np.array_equal(b,np.array([[1,2],[3,4]], dtype = np.float32)))
    

def can_convert_numpy_Interval_array_to_lower():
    """
    test function get_lower for numpy interval array
    """
    # numpy interval case
    a = np.array([[Interval(1,1), Interval(1,2)], [Interval(1,3), Interval(1,4)]])
    b = get_lower(a, ret = 'torch')
    assert(b.dtype == torch.float32)
    assert(torch.equal(b,torch.tensor([[1,1],[1,1]], dtype = torch.float32)))


def can_convert_torch_pseudo_interval_array_to_upper():
    """
    test function get_upper for torch pseudo interval array
    """
    # torch pseudo interval case
    a = torch.tensor([[1,2,3,4], [5,6,7,8]], dtype = torch.float32, device='cuda')
    b = get_upper(a)
    assert(b.dtype == a.dtype)
    assert(b.device == torch.device('cpu'))
    assert(torch.equal(b,torch.tensor([[2,4],[6,8]], dtype = torch.float32)))


def can_convert_torch_pseudo_interval_array_to_lower():
    """
    test function get_lower for torch pseudo interval array
    """
    # torch pseudo interval case
    a = torch.tensor([[1,2,3,4], [5,6,7,8]], dtype = torch.float32, device='cuda')
    b = get_lower(a)
    assert(b.dtype == a.dtype)
    assert(b.device == torch.device('cpu'))
    assert(torch.equal(b,torch.tensor([[1,3],[5,7]], dtype = torch.float32)))


def add_test():
    """
    test add api for all 4 cases
    """
    # numpy float
    A = np.array([[1,2],[3,4],[5,6]], dtype = np.float32)
    B = np.ones((3,2), dtype = np.float32)
    res = add(A, B, False)
    assert(np.array_equal(res, np.array([[2,3],[4,5],[6,7]], dtype = np.float32)))
    # torch float
    A = torch.tensor([[1,2],[3,4],[5,6]], dtype = torch.float32, device = 'cuda')
    B = torch.ones((3,2), dtype = torch.float32, device = 'cuda')
    res = add(A, B, False)
    assert(torch.equal(res, torch.tensor([[2,3],[4,5],[6,7]], dtype = torch.float32, device = 'cuda')))
    # numpy interval
    A = np.array([Interval(1,2), Interval(3,4)], dtype= Interval)
    B = np.array([Interval(0,1), Interval(5,6)], dtype= Interval)
    res = add(A, B, True)
    assert(res.dtype == object)
    assert(np.array_equal(res, np.array([Interval(nextafter(1, -inf),nextafter(3, inf)), Interval(nextafter(8, -inf),nextafter(10, inf))], dtype=Interval)))
    # torch interval
    A = torch.tensor([[1,2],[3,4]], dtype=torch.float32, device = 'cuda')
    B = torch.zeros((2,2), dtype=torch.float32, device = 'cuda')
    res = add(A,B,True)
    assert(torch.equal(res, torch.tensor([[1,nextafter(2, inf)],[nextafter(3, -inf),nextafter(4, inf)]], dtype=torch.float32, device = 'cuda')))
    

def mat_mul_test():
    """
    test mat_mul api for all 4 cases
    """
    # numpy float
    A = np.array([[1,2,3],[4,5,6]], dtype = np.float32)
    B = np.ones((3,2), dtype = np.float32)
    res = mat_mul(A, B, False)
    assert(np.array_equal(res, np.array([[6,6],[15,15]], dtype = np.float32)))
    # torch float
    A = torch.tensor([[1,2,3],[4,5,6]], dtype = torch.float32, device='cuda')
    B = torch.ones((3,2), dtype = torch.float32, device='cuda')
    res = mat_mul(A, B, False)
    assert(torch.equal(res, torch.tensor([[6,6],[15,15]], dtype = torch.float32, device='cuda')))
    # numpy interval 
    A = np.array([[Interval(1,2), Interval(-2,3)]], dtype= Interval)
    B = np.array([[Interval(-3,4)], [Interval(5,6)]], dtype= Interval)
    res = mat_mul(A, B, True) # should be around (-18,26)
    assert(res.dtype == object)
    assert(res[0][0].lower <= -18 and res[0][0].upper >= 26)
    # torch interval
    A = torch.tensor([[1,2,-2,3]], dtype= torch.float32, device = 'cuda')
    B = torch.tensor([[-3,4], [5,6]], dtype= torch.float32, device='cuda')
    res = mat_mul(A, B, True) # should be around (-18,26)
    assert(res.dtype==torch.float32)
    assert(res[0][0] <= -18 and res[0][1] >= 26)


def gemm_test():
    """
    test gemm api for all 4 cases
    """
    # numpy float
    A = np.array([[1,2,3],[4,5,6]], dtype = np.float32)
    B = np.ones((3,2), dtype = np.float32)
    bias = np.ones((2,2), dtype = np.float32)
    res = gemm(A, B, bias, False)
    assert(np.array_equal(res, np.array([[7,7],[16,16]], dtype = np.float32)))
    # torch float
    A = torch.tensor([[1,2,3],[4,5,6]], dtype = torch.float32, device='cuda')
    B = torch.ones((3,2), dtype = torch.float32, device='cuda')
    bias = torch.ones((2,2), dtype = torch.float32, device='cuda')
    res = gemm(A, B, bias, False)
    assert(torch.equal(res, torch.tensor([[7,7],[16,16]], dtype = torch.float32, device='cuda')))
    # numpy interval 
    A = np.array([[Interval(1,2), Interval(-2,3)]], dtype= Interval)
    B = np.array([[Interval(-3,4)], [Interval(5,6)]], dtype= Interval)
    bias = np.array([[Interval(-1,1)]], dtype= Interval)
    res = gemm(A, B, bias, True) # should be around (-19,27)
    assert(res.dtype == object)
    assert(res[0][0].lower <= -19 and res[0][0].upper >= 27)
    # torch interval
    A = torch.tensor([[1,2,-2,3]], dtype= torch.float32, device = 'cuda')
    B = torch.tensor([[-3,4], [5,6]], dtype= torch.float32, device='cuda')
    bias = torch.tensor([[-1,1]], dtype= torch.float32, device='cuda')
    res = gemm(A, B, bias, True) # should be around (-19,27)
    assert(res.dtype==torch.float32)
    assert(res[0][0] <= -19 and res[0][1] >= 27)


def gemm_calculation_speed_test():
    """
    test gemm calculation speed for all 4 cases under different matrix size
    result is written to a local csv file
    """
    header = ["matrix_size", "numpy_float(s)", "torch_float(s)", "numpy_interval(s)", "torch_interval(s)"]
    all_content = []
    for i in range(20, 201):
        row = []
        row.append(i)
        # numpy float
        A_np_float = np.random.rand(i,i)
        B_np_float = np.random.rand(i,i)
        bias_np_float = np.random.rand(i,i)
        start = time.time()
        res = gemm(A_np_float,B_np_float,bias_np_float,False)
        duration = time.time() - start
        row.append(f"{duration:.6f}")
        # torch float
        A_torch_float = torch.tensor(A_np_float, dtype=torch.float32, device='cuda')
        B_torch_float = torch.tensor(B_np_float, dtype=torch.float32, device='cuda')
        bias_torch_float = torch.tensor(bias_np_float, dtype=torch.float32, device='cuda')
        start = time.time()
        res = gemm(A_torch_float, B_torch_float, bias_torch_float,False)
        duration = time.time() - start
        row.append(f"{duration:.6f}")
        # numpy interval
        A_np_interval = float_array2np_interval_cpu_array(A_np_float)
        B_np_interval = float_array2np_interval_cpu_array(B_np_float)
        bias_np_interval = float_array2np_interval_cpu_array(bias_np_float)
        start = time.time()
        res = gemm(A_np_interval,B_np_interval,bias_np_interval,True)
        duration = time.time() - start
        row.append(f"{duration:.6f}")
        # torch interval
        A_torch_float = torch.tensor(np.random.rand(i,2*i), dtype=torch.float32, device='cuda')
        B_torch_float = torch.tensor(np.random.rand(i,2*i), dtype=torch.float32, device='cuda')
        bias_torch_float = torch.tensor(np.random.rand(i,2*i), dtype=torch.float32, device='cuda')
        start = time.time()
        res = gemm(A_torch_float, B_torch_float, bias_torch_float,True)
        duration = time.time() - start
        row.append(f"{duration:.6f}")
        all_content.append(row)

    with open("gemm_calculation_speed_test.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_content)


if __name__ == '__main__':
    can_convert_real_numbered_np_array_to_Interval_np_array()
    can_convert_float_torch_array_to_pesudo_interval_array()
    can_convert_numpy_Interval_array_to_upper()
    can_convert_numpy_Interval_array_to_lower()
    can_convert_torch_pseudo_interval_array_to_upper()
    can_convert_torch_pseudo_interval_array_to_lower()
    add_test()
    mat_mul_test()
    gemm_test()
    # gemm_calculation_speed_test()