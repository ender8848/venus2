Some test configurationsdoes not work, this file records some problematic configurations and outputs.

# Peoblems in acasxu test

When running tests using the following command: 

```
python3 __main__.py --net ./venus/compatible_tests/data/acasxu/nets/ --spec ./venus/compatible_tests/data/acasxu/specs/ --timeout 60
```

The following error appears (It seems to fail in all complete methods where gurobi is needed): 

```sh
Verifying ACASXU_run2a_1_1_batch_2000.onnx against prop_4.vnnlib
Traceback (most recent call last):
  File "/home/hao/Documents/venus2/__main__.py", line 171, in <module>
    main()
  File "/home/hao/Documents/venus2/__main__.py", line 168, in main
    venus.verify()
  File "/home/hao/Documents/venus2/venus/verification/venus.py", line 91, in verify
    ver_report = self.verify_query(nn, spec)
  File "/home/hao/Documents/venus2/venus/verification/venus.py", line 161, in verify_query
    sub_ver_report = verifier.verify()
  File "/home/hao/Documents/venus2/venus/verification/verifier.py", line 64, in verify
    ver_report = self.verify_complete()
  File "/home/hao/Documents/venus2/venus/verification/verifier.py", line 127, in verify_complete
    report = self.verify_incomplete()
  File "/home/hao/Documents/venus2/venus/verification/verifier.py", line 87, in verify_incomplete
    cex = pgd.start(self.prob)
  File "/home/hao/Documents/venus2/venus/verification/pgd.py", line 39, in start
    distribution = torch.distributions.uniform.Uniform(
  File "/home/hao/.local/share/virtualenvs/venus2-wTg3y2Yr/lib/python3.10/site-packages/torch/distributions/uniform.py", line 56, in __init__
    raise ValueError("Uniform is not defined when low>= high")
ValueError: Uniform is not defined when low>= high
```

```sh
Verifying ACASXU_run2a_1_1_batch_2000.onnx against prop_6.vnnlib...:   1%|▏                  | 4/405 [00:05<08:31,  1.27s/it]
Traceback (most recent call last):
  File "/home/hao/Documents/venus2/__main__.py", line 171, in <module>
    main()
  File "/home/hao/Documents/venus2/__main__.py", line 168, in main
    venus.verify()
  File "/home/hao/Documents/venus2/venus/verification/venus.py", line 79, in verify
    spec = vnn_parser.parse()
  File "/home/hao/Documents/venus2/venus/input/vnnlib_parser.py", line 66, in parse
    Input(clause[0][0], clause[0][1]),
  File "/home/hao/Documents/venus2/venus/network/node.py", line 235, in __init__
    bounds.lower.shape,
AttributeError: 'Tensor' object has no attribute 'lower'
```

Then I removed the specs prop_4 and prop_6, it works fine.


# Problems in mnistfc test

When running tests using the following command: 

```
python3 __main__.py --net ./venus/compatible_tests/data/mnistfc/nets/ --spec ./venus/compatible_tests/data/mnistfc/specs/0.03/ --timeout 60
```

Sometimes there are gurobipy.GurobiError. Error message is same in different verification problems. 

```sh
File "/home/hao/Documents/venus2/venus/verification/verification_process.py", line 66, in run
    slv_report = self.complete_ver(prob)
  File "/home/hao/Documents/venus2/venus/verification/verification_process.py", line 87, in complete_ver
    return  slv.solve()
  File "/home/hao/Documents/venus2/venus/solver/milp_solver.py", line 86, in solve
    gmodel.optimize(MILPSolver._callback)
  File "src/gurobipy/model.pxi", line 864, in gurobipy.Model.optimize
gurobipy.GurobiError
```