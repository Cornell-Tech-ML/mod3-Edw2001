# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## Task 3_1 & Task 3_2
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, D:\Cornell
 Tech\2024 Fall\Machine Learning
Engineering\workspace\mod3-Edw2001\minitorch\fast_ops.py (163)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, D:\Cornell Tech\2024 Fall\Machine Learning Engineering\workspace\mod3-Edw2001\minitorch\fast_ops.py (163)
-------------------------------------------------------------------------|loop #ID
    def _map(                                                            |
        out: Storage,                                                    |
        out_shape: Shape,                                                |
        out_strides: Strides,                                            |
        in_storage: Storage,                                             |
        in_shape: Shape,                                                 |
        in_strides: Strides,                                             |
    ) -> None:                                                           |
        # TODO: Implement for Task 3.1.                                  |
        out_index = np.empty(MAX_DIMS, dtype=np.int32)                   |
        in_index = np.empty(MAX_DIMS, dtype=np.int32)                    |
                                                                         |
        for out_flat_index  in prange(len(out)):-------------------------| #0
            out_index = np.empty(MAX_DIMS, np.int32)                     |
            in_index = np.empty(MAX_DIMS, np.int32)                      |
            to_index(out_flat_index, out_shape, out_index)               |
            broadcast_index(out_index, out_shape, in_shape, in_index)    |
            out_pos = index_to_position(out_index, out_strides)          |
            int_pos = index_to_position(in_index, in_strides)            |
            out[out_pos] = fn(in_storage[int_pos])                       |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #0).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at D:\Cornell Tech\2024
Fall\Machine Learning Engineering\workspace\mod3-Edw2001\minitorch\fast_ops.py
(176) is hoisted out of the parallel loop labelled #0 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at D:\Cornell Tech\2024
Fall\Machine Learning Engineering\workspace\mod3-Edw2001\minitorch\fast_ops.py
(177) is hoisted out of the parallel loop labelled #0 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: in_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, D:\Cornell
 Tech\2024 Fall\Machine Learning
Engineering\workspace\mod3-Edw2001\minitorch\fast_ops.py (211)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, D:\Cornell Tech\2024 Fall\Machine Learning Engineering\workspace\mod3-Edw2001\minitorch\fast_ops.py (211)
------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                       |
        out: Storage,                                                               |
        out_shape: Shape,                                                           |
        out_strides: Strides,                                                       |
        a_storage: Storage,                                                         |
        a_shape: Shape,                                                             |
        a_strides: Strides,                                                         |
        b_storage: Storage,                                                         |
        b_shape: Shape,                                                             |
        b_strides: Strides,                                                         |
    ) -> None:                                                                      |
        # TODO: Implement for Task 3.1.                                             |
        for out_flat_index in prange(len(out)):-------------------------------------| #1
            out_index = np.empty(MAX_DIMS, np.int32)                                |
            a_index = np.empty(MAX_DIMS, np.int32)                                  |
            b_index = np.empty(MAX_DIMS, np.int32)                                  |
            to_index(out_flat_index, out_shape, out_index)                          |
            out_flat_pos = index_to_position(out_index, out_strides)                |
            # Broadcast indices for tensor a and b                                  |
            broadcast_index(out_index, out_shape, a_shape, a_index)                 |
            broadcast_index(out_index, out_shape, b_shape, b_index)                 |
                                                                                    |
            a_flat_pos = index_to_position(a_index, a_strides)                      |
            b_flat_pos = index_to_position(b_index, b_strides)                      |
            out[out_flat_pos] = fn(a_storage[a_flat_pos], b_storage[b_flat_pos])    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at D:\Cornell Tech\2024
Fall\Machine Learning Engineering\workspace\mod3-Edw2001\minitorch\fast_ops.py
(224) is hoisted out of the parallel loop labelled #1 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at D:\Cornell Tech\2024
Fall\Machine Learning Engineering\workspace\mod3-Edw2001\minitorch\fast_ops.py
(225) is hoisted out of the parallel loop labelled #1 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at D:\Cornell Tech\2024
Fall\Machine Learning Engineering\workspace\mod3-Edw2001\minitorch\fast_ops.py
(226) is hoisted out of the parallel loop labelled #1 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
D:\Cornell Tech\2024 Fall\Machine Learning
Engineering\workspace\mod3-Edw2001\minitorch\fast_ops.py (261)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, D:\Cornell Tech\2024 Fall\Machine Learning Engineering\workspace\mod3-Edw2001\minitorch\fast_ops.py (261)
------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                              |
        out: Storage,                                                         |
        out_shape: Shape,                                                     |
        out_strides: Strides,                                                 |
        a_storage: Storage,                                                   |
        a_shape: Shape,                                                       |
        a_strides: Strides,                                                   |
        reduce_dim: int,                                                      |
    ) -> None:                                                                |
        # TODO: Implement for Task 3.1.                                       |
                                                                              |
        for out_flat_index in prange(len(out)):-------------------------------| #2
            out_multi_index = np.empty(MAX_DIMS, np.int32)                    |
            to_index(out_flat_index, out_shape, out_multi_index)              |
            out_pos = index_to_position(out_multi_index, out_strides)         |
            for reduce_index in range(a_shape[reduce_dim]):                   |
                out_multi_index[reduce_dim] = reduce_index                    |
                a_flat_pos = index_to_position(out_multi_index, a_strides)    |
                out[out_pos] = fn(out[out_pos], a_storage[a_flat_pos])        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #2).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at D:\Cornell Tech\2024
Fall\Machine Learning Engineering\workspace\mod3-Edw2001\minitorch\fast_ops.py
(273) is hoisted out of the parallel loop labelled #2 (it will be performed
before the loop is executed and reused inside the loop):
   Allocation:: out_multi_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, D:\Cornell
Tech\2024 Fall\Machine Learning
Engineering\workspace\mod3-Edw2001\minitorch\fast_ops.py (285)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, D:\Cornell Tech\2024 Fall\Machine Learning Engineering\workspace\mod3-Edw2001\minitorch\fast_ops.py (285)
--------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                      |
    out: Storage,                                                                                 |
    out_shape: Shape,                                                                             |
    out_strides: Strides,                                                                         |
    a_storage: Storage,                                                                           |
    a_shape: Shape,                                                                               |
    a_strides: Strides,                                                                           |
    b_storage: Storage,                                                                           |
    b_shape: Shape,                                                                               |
    b_strides: Strides,                                                                           |
) -> None:                                                                                        |
    """NUMBA tensor matrix multiply function.                                                     |
                                                                                                  |
    Should work for any tensor shapes that broadcast as long as                                   |
                                                                                                  |
    ```                                                                                           |
    assert a_shape[-1] == b_shape[-2]                                                             |
    ```                                                                                           |
                                                                                                  |
    Optimizations:                                                                                |
                                                                                                  |
    * Outer loop in parallel                                                                      |
    * No index buffers or function calls                                                          |
    * Inner loop should have no global writes, 1 multiply.                                        |
                                                                                                  |
                                                                                                  |
    Args:                                                                                         |
    ----                                                                                          |
        out (Storage): storage for `out` tensor                                                   |
        out_shape (Shape): shape for `out` tensor                                                 |
        out_strides (Strides): strides for `out` tensor                                           |
        a_storage (Storage): storage for `a` tensor                                               |
        a_shape (Shape): shape for `a` tensor                                                     |
        a_strides (Strides): strides for `a` tensor                                               |
        b_storage (Storage): storage for `b` tensor                                               |
        b_shape (Shape): shape for `b` tensor                                                     |
        b_strides (Strides): strides for `b` tensor                                               |
                                                                                                  |
    Returns:                                                                                      |
    -------                                                                                       |
        None : Fills in `out`                                                                     |
                                                                                                  |
    """                                                                                           |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                        |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                        |
                                                                                                  |
    # TODO: Implement for Task 3.2.                                                               |
    assert a_shape[-1] == b_shape[-2]                                                             |
                                                                                                  |
    for batch in prange(out_shape[0]):------------------------------------------------------------| #3
        for row in range(out_shape[1]):                                                           |
            for col in range(out_shape[2]):                                                       |
                accumulator = 0.0                                                                 |
                a_idx = batch * a_batch_stride + row * a_strides[1]                               |
                b_idx = batch * b_batch_stride + col * b_strides[2]                               |
                                                                                                  |
                # dot product`                                                                    |
                for k in range(a_shape[2]):                                                       |
                    accumulator += a_storage[a_idx] * b_storage[b_idx]                            |
                    a_idx += a_strides[2]  # Move to the next element in the row of `a`           |
                    b_idx += b_strides[1]  # Move to the next element in the column of `b`        |
                                                                                                  |
                out_idx = batch * out_strides[0] + row * out_strides[1] + col * out_strides[2]    |
                out[out_idx] = accumulator                                                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None

## Split dataset, GPU, Hidden 100, Rate 0.05
Epoch  0  loss  8.043145030319518 correct 30
Epoch  10  loss  9.7558529617398 correct 46
Epoch  20  loss  5.286052986850942 correct 47
Epoch  30  loss  3.631971759032841 correct 49
Epoch  40  loss  2.663483312852689 correct 49
Epoch  50  loss  2.9236070970684613 correct 45
Epoch  60  loss  3.0061213783326206 correct 50
Epoch  70  loss  2.0117684895259846 correct 50
Epoch  80  loss  1.6311887367806046 correct 46
Epoch  90  loss  3.199825980441876 correct 47
Epoch  100  loss  1.9043237663859254 correct 50
Epoch  110  loss  0.6423647969614927 correct 49
Epoch  120  loss  2.463445246064017 correct 44
Epoch  130  loss  1.6035408539262619 correct 50
Epoch  140  loss  0.328132403588011 correct 50
Epoch  150  loss  1.2169674172272686 correct 50
Epoch  160  loss  0.45487332990705814 correct 50
Epoch  170  loss  1.2560121263422117 correct 50
Epoch  180  loss  0.5596979939004286 correct 50
Epoch  190  loss  1.0275914628769063 correct 50
Epoch  200  loss  0.7054071878015293 correct 50
Epoch  210  loss  0.24240767629326596 correct 50
Epoch  220  loss  1.436220455694159 correct 50
Epoch  230  loss  0.8185612626810846 correct 50
Epoch  240  loss  0.4604459724486232 correct 50
Epoch  250  loss  0.6161905796438899 correct 50
Epoch  260  loss  0.45936669217590187 correct 50
Epoch  270  loss  0.31515643941971355 correct 50
Epoch  280  loss  0.09613625134354849 correct 50
Epoch  290  loss  0.760081404057984 correct 50
Epoch  300  loss  0.6761951422859962 correct 50
Epoch  310  loss  0.5488133727740518 correct 50
Epoch  320  loss  0.35279684503543196 correct 50
Epoch  330  loss  0.20091989598529775 correct 50
Epoch  340  loss  0.4710526094537106 correct 50
Epoch  350  loss  0.2648111387259521 correct 50
Epoch  360  loss  0.18721152023707197 correct 50
Epoch  370  loss  0.29794609126697125 correct 50
Epoch  380  loss  0.3724872765507409 correct 50
Epoch  390  loss  0.21620827861808384 correct 50
Epoch  400  loss  0.30949379614122646 correct 50
Epoch  410  loss  0.20793879360772835 correct 50
Epoch  420  loss  0.2952754782612684 correct 50
Epoch  430  loss  0.1835355910435392 correct 50
Epoch  440  loss  0.05590274903630034 correct 50
Epoch  450  loss  0.14140203053618067 correct 50
Epoch  460  loss  0.11487049627091106 correct 50
Epoch  470  loss  0.22474568261635333 correct 50
Epoch  480  loss  0.28804592928275236 correct 50
Epoch  490  loss  0.04352535743468793 correct 50

## Xor dataset, GPU, Hidden 100, Rate 0.05
Epoch  0  loss  5.2864295768354035 correct 30
Epoch  10  loss  5.959720900494091 correct 41
Epoch  20  loss  4.572801023529927 correct 41
Epoch  30  loss  2.1933217413214563 correct 42
Epoch  40  loss  4.067211230964851 correct 45
Epoch  50  loss  2.5792381873882566 correct 43
Epoch  60  loss  1.3878658629184772 correct 46
Epoch  70  loss  3.112042017520589 correct 46
Epoch  80  loss  2.021699690993473 correct 47
Epoch  90  loss  2.0590948994631653 correct 46
Epoch  100  loss  2.9378733653755007 correct 45
Epoch  110  loss  1.1671776182549758 correct 48
Epoch  120  loss  2.9948125124348324 correct 45
Epoch  130  loss  1.8224821349317446 correct 46
Epoch  140  loss  1.2388280128867237 correct 48
Epoch  150  loss  4.053160935806033 correct 47
Epoch  160  loss  0.502675644194002 correct 48
Epoch  170  loss  3.3940031688655816 correct 48
Epoch  180  loss  1.6457487029583986 correct 48
Epoch  190  loss  0.9875857906337411 correct 48
Epoch  200  loss  2.735685302282098 correct 49
Epoch  210  loss  0.5187883833967784 correct 48
Epoch  220  loss  1.6915569816740896 correct 48
Epoch  230  loss  2.37524138003681 correct 48
Epoch  240  loss  0.6049461266711784 correct 48
Epoch  250  loss  1.2957568858863207 correct 48
Epoch  260  loss  1.616940419769501 correct 49
Epoch  270  loss  0.16733686363446082 correct 48
Epoch  280  loss  2.542333772909461 correct 47
Epoch  290  loss  0.4768189486978905 correct 48
Epoch  300  loss  0.32661363216508865 correct 48
Epoch  310  loss  2.801444708826679 correct 47
Epoch  320  loss  0.21147512009234096 correct 48
Epoch  330  loss  0.20590200019885674 correct 49
Epoch  340  loss  0.512457670050257 correct 48
Epoch  350  loss  1.2981902597234303 correct 48
Epoch  360  loss  1.4756076189422354 correct 47
Epoch  370  loss  1.3169035597741847 correct 48
Epoch  380  loss  0.26182133103272054 correct 48
Epoch  390  loss  1.6741901413404976 correct 48
Epoch  400  loss  2.1524796091473104 correct 48
Epoch  410  loss  0.6867089615747773 correct 48
Epoch  420  loss  0.4388629636137098 correct 48
Epoch  430  loss  1.8403882831341982 correct 48
Epoch  440  loss  0.23387722628258314 correct 49
Epoch  450  loss  2.2009111872761427 correct 49
Epoch  460  loss  0.9412643915759972 correct 49
Epoch  470  loss  0.5631852076333177 correct 48
Epoch  480  loss  0.621903191307054 correct 48
Epoch  490  loss  1.0930475470881353 correct 49

## Simple dataset, GPU, Hidden 100, Rate 0.05
Epoch  0  loss  6.404937102089181 correct 48
Epoch  10  loss  0.8005045882504618 correct 49
Epoch  20  loss  1.199842678126724 correct 49
Epoch  30  loss  0.7312497887560522 correct 49
Epoch  40  loss  0.6941775889708987 correct 49
Epoch  50  loss  0.5542845272396043 correct 49
Epoch  60  loss  0.34692173747618443 correct 50
Epoch  70  loss  0.9069422918890245 correct 50
Epoch  80  loss  0.4483088167396214 correct 50
Epoch  90  loss  0.69546134512032 correct 50
Epoch  100  loss  0.02669815894369867 correct 50
Epoch  110  loss  0.3635890213560857 correct 50
Epoch  120  loss  0.24222100918937958 correct 50
Epoch  130  loss  0.051614878704503805 correct 50
Epoch  140  loss  0.22139894746927014 correct 50
Epoch  150  loss  0.17041821460393328 correct 50
Epoch  160  loss  0.03375380803483051 correct 50
Epoch  170  loss  0.35631119078767265 correct 50
Epoch  180  loss  0.24572329822105332 correct 50
Epoch  190  loss  0.000647301553613737 correct 50
Epoch  200  loss  0.3126368414046676 correct 50
Epoch  210  loss  0.20648788678574462 correct 50
Epoch  220  loss  0.001628957377101976 correct 50
Epoch  230  loss  0.11665995310710835 correct 50
Epoch  240  loss  0.0956282640780763 correct 50
Epoch  250  loss  0.0065519714687951 correct 50
Epoch  260  loss  0.02726644083226521 correct 50
Epoch  270  loss  0.2981978700552636 correct 50
Epoch  280  loss  0.30219675161181664 correct 50
Epoch  290  loss  0.018880313061733694 correct 50
Epoch  300  loss  0.1573529167769306 correct 50
Epoch  310  loss  0.12673677547436246 correct 50
Epoch  320  loss  0.005852797822004353 correct 50
Epoch  330  loss  0.09627406477468244 correct 50
Epoch  340  loss  0.2135052073106856 correct 50
Epoch  350  loss  0.18577248661668927 correct 50
Epoch  360  loss  0.0017231154708638786 correct 50
Epoch  370  loss  0.1619904990763875 correct 50
Epoch  380  loss  0.06996236519482807 correct 50
Epoch  390  loss  0.07670800248439645 correct 50
Epoch  400  loss  0.0003655857315467424 correct 50
Epoch  410  loss  0.0014463963120550831 correct 50
Epoch  420  loss  0.05810583626637832 correct 50
Epoch  430  loss  0.0013240503502694987 correct 50
Epoch  440  loss  0.07956342048084422 correct 50
Epoch  450  loss  0.1916933484131694 correct 50
Epoch  460  loss  0.08914171405560464 correct 50
Epoch  470  loss  0.1998203844622885 correct 50
Epoch  480  loss  0.06752154965784254 correct 50
Epoch  490  loss  0.12086282067693582 correct 50

## Split dataset, CPU, Hidden 100, Rate 0.05
Epoch  0  loss  8.218459649408432 correct 29
Epoch  10  loss  7.357064994425451 correct 31
Epoch  20  loss  4.717944503803417 correct 48
Epoch  30  loss  5.2555680866486 correct 47
Epoch  40  loss  3.390753315970908 correct 46
Epoch  50  loss  2.071198532707255 correct 49
Epoch  60  loss  3.0122363511757766 correct 46
Epoch  70  loss  1.5993689414539698 correct 49
Epoch  80  loss  1.2240118854818256 correct 49
Epoch  90  loss  1.4352247986375644 correct 49
Epoch  100  loss  1.4880279808680927 correct 49
Epoch  110  loss  0.9507903862715615 correct 49
Epoch  120  loss  0.9415029465625568 correct 49
Epoch  130  loss  1.202121476169289 correct 49
Epoch  140  loss  1.587362930841988 correct 50
Epoch  150  loss  0.3658822303195507 correct 49
Epoch  160  loss  0.7908159080057381 correct 50
Epoch  170  loss  0.3799926923662427 correct 50
Epoch  180  loss  0.5238231593662699 correct 49
Epoch  190  loss  1.1317255537681743 correct 49
Epoch  200  loss  1.09941443758914 correct 50
Epoch  210  loss  0.47280890854813207 correct 50
Epoch  220  loss  0.9602027987027365 correct 50
Epoch  230  loss  0.7636031100035663 correct 50
Epoch  240  loss  0.2764065972593755 correct 50
Epoch  250  loss  0.43923834968520126 correct 50
Epoch  260  loss  0.20240362761065053 correct 49
Epoch  270  loss  0.14579881858789817 correct 50
Epoch  280  loss  0.05681892616043129 correct 49
Epoch  290  loss  1.0813136412063622 correct 50
Epoch  300  loss  0.9027409704422481 correct 50
Epoch  310  loss  0.39470224409790605 correct 50
Epoch  320  loss  0.3075959833489313 correct 50
Epoch  330  loss  0.22893170566272852 correct 49
Epoch  340  loss  0.0677156074311148 correct 50
Epoch  350  loss  0.5205824297421487 correct 49
Epoch  360  loss  1.0309821826248176 correct 49
Epoch  370  loss  0.15582845297024822 correct 49
Epoch  380  loss  0.027727473530953978 correct 50
Epoch  390  loss  0.824909895881654 correct 50
Epoch  400  loss  0.2944611882463158 correct 50
Epoch  410  loss  0.1401348658291991 correct 50
Epoch  420  loss  0.4989109687404394 correct 50
Epoch  430  loss  0.04178198398804933 correct 50
Epoch  440  loss  0.41938248007451323 correct 49
Epoch  450  loss  0.9924359407508101 correct 50
Epoch  460  loss  0.4988711616753034 correct 49
Epoch  470  loss  0.05639270808329331 correct 50
Epoch  480  loss  0.7502674225783234 correct 49
Epoch  490  loss  0.11033938455433757 correct 50

## Xor dataset, CPU, Hidden 100, Rate 0.05
Epoch  0  loss  5.811767106877591 correct 33
Epoch  10  loss  4.6589281733986665 correct 46
Epoch  20  loss  3.202814546018072 correct 45
Epoch  30  loss  4.076228216920744 correct 45
Epoch  40  loss  2.698517533525783 correct 47
Epoch  50  loss  3.3939739243929634 correct 47
Epoch  60  loss  2.811606966757771 correct 48
Epoch  70  loss  2.3041460825776316 correct 47
Epoch  80  loss  1.5789945156717669 correct 47
Epoch  90  loss  2.955228608635249 correct 47
Epoch  100  loss  4.232438628889724 correct 46
Epoch  110  loss  0.7548518890127514 correct 49
Epoch  120  loss  0.7824922239320475 correct 49
Epoch  130  loss  1.9545160901348215 correct 49
Epoch  140  loss  2.5524184851894387 correct 49
Epoch  150  loss  1.5907034049003022 correct 49
Epoch  160  loss  1.9324249202211565 correct 50
Epoch  170  loss  2.333863675765558 correct 49
Epoch  180  loss  0.25236204109401766 correct 49
Epoch  190  loss  0.5321818563081774 correct 49
Epoch  200  loss  1.034353385898915 correct 49
Epoch  210  loss  1.3601492438663778 correct 50
Epoch  220  loss  1.5279785404274775 correct 49
Epoch  230  loss  1.3380554817147101 correct 50
Epoch  240  loss  0.6299413065087768 correct 49
Epoch  250  loss  1.4625537065626495 correct 50
Epoch  260  loss  0.47604737002851133 correct 50
Epoch  270  loss  1.363659367446051 correct 50
Epoch  280  loss  1.027672335636676 correct 49
Epoch  290  loss  0.8778115794938988 correct 50
Epoch  300  loss  0.6057078616112395 correct 50
Epoch  310  loss  0.7824430697029706 correct 50
Epoch  320  loss  0.5142418493485439 correct 50
Epoch  330  loss  0.7104445067122563 correct 50
Epoch  340  loss  1.291974996415675 correct 50
Epoch  350  loss  0.17209612436490915 correct 50
Epoch  360  loss  0.1620900492822562 correct 48
Epoch  370  loss  0.35399645095578813 correct 50
Epoch  380  loss  0.38437158545540245 correct 50
Epoch  390  loss  1.415672331711011 correct 50
Epoch  400  loss  0.3995659917772082 correct 50
Epoch  410  loss  0.2079560693943991 correct 50
Epoch  420  loss  0.5671087936588612 correct 50
Epoch  430  loss  0.4470766659218396 correct 50
Epoch  440  loss  0.33331759958312607 correct 50
Epoch  450  loss  0.7649339609313789 correct 50
Epoch  460  loss  0.46510696963753756 correct 50
Epoch  470  loss  1.0388638717964533 correct 50
Epoch  480  loss  1.0957954643723298 correct 50
Epoch  490  loss  0.5154081631542755 correct 50

## Simple dataset, CPU, Hidden 100, Rate 0.05
Epoch  0  loss  5.430510163022588 correct 48
Epoch  10  loss  1.2153114132503862 correct 49
Epoch  20  loss  1.277729505761822 correct 50
Epoch  30  loss  0.5396918632043975 correct 50
Epoch  40  loss  0.8859311383739744 correct 50
Epoch  50  loss  0.287604417954165 correct 50
Epoch  60  loss  1.178530945435418 correct 50
Epoch  70  loss  0.8512486853589926 correct 50
Epoch  80  loss  0.4720381165210195 correct 50
Epoch  90  loss  0.746064170605783 correct 50
Epoch  100  loss  0.007926338429794089 correct 50
Epoch  110  loss  0.2650320121110456 correct 50
Epoch  120  loss  0.020489400012374655 correct 50
Epoch  130  loss  0.6514262687314119 correct 50
Epoch  140  loss  0.7037071745775202 correct 50
Epoch  150  loss  0.08339362060076967 correct 50
Epoch  160  loss  0.003060886225119097 correct 50
Epoch  170  loss  0.26834330558150277 correct 50
Epoch  180  loss  0.2002893876864961 correct 50
Epoch  190  loss  0.047269900219063854 correct 50
Epoch  200  loss  0.2851796413802973 correct 50
Epoch  210  loss  0.35709359220991554 correct 50
Epoch  220  loss  0.059130763996095397 correct 50
Epoch  230  loss  0.3934144186216887 correct 50
Epoch  240  loss  0.43184582222688084 correct 50
Epoch  250  loss  0.4916527914753972 correct 50
Epoch  260  loss  0.047099946066565236 correct 50
Epoch  270  loss  0.0003345442311317912 correct 50
Epoch  280  loss  0.4470814769230023 correct 50
Epoch  290  loss  0.02547800456820043 correct 50
Epoch  300  loss  0.014690917856152763 correct 50
Epoch  310  loss  0.015660847109515937 correct 50
Epoch  320  loss  0.044371136069761306 correct 50
Epoch  330  loss  0.08238274789140337 correct 50
Epoch  340  loss  0.05250846384292394 correct 50
Epoch  350  loss  0.040038686028307256 correct 50
Epoch  360  loss  0.03990620321148004 correct 50
Epoch  370  loss  0.10339443492460682 correct 50
Epoch  380  loss  0.06884464149147601 correct 50
Epoch  390  loss  0.00016821334007506968 correct 50
Epoch  400  loss  7.736174859544921e-05 correct 50
Epoch  410  loss  0.002142354051336183 correct 50
Epoch  420  loss  0.2562171518772726 correct 50
Epoch  430  loss  0.03581328927452937 correct 50
Epoch  440  loss  2.124670370442157e-05 correct 50
Epoch  450  loss  0.02268424964312535 correct 50
Epoch  460  loss  0.0012468705636287237 correct 50
Epoch  470  loss  0.32007695613115783 correct 50
Epoch  480  loss  0.07088710121944082 correct 50
Epoch  490  loss  0.0008055020737165872 correct 50

## Split dataset, GPU, Hidden 200, Rate 0.05
Epoch  0  loss  16.656355579735973 correct 35
Epoch  10  loss  2.1424484592638784 correct 44
Epoch  20  loss  2.1936725437667386 correct 45
Epoch  30  loss  3.302188722194166 correct 42
Epoch  40  loss  2.362841496723594 correct 46
Epoch  50  loss  1.3174790745958183 correct 48
Epoch  60  loss  1.0631289750387436 correct 48
Epoch  70  loss  0.3939252585873045 correct 46
Epoch  80  loss  1.2573689243721353 correct 46
Epoch  90  loss  0.4876414648117575 correct 48
Epoch  100  loss  0.30751066668844707 correct 48
Epoch  110  loss  1.2117615811694942 correct 48
Epoch  120  loss  0.7770530714450553 correct 49
Epoch  130  loss  1.8021768824199857 correct 48
Epoch  140  loss  2.445544654506333 correct 48
Epoch  150  loss  1.278415846723239 correct 49
Epoch  160  loss  1.4727109653998698 correct 49
Epoch  170  loss  0.35643561215229347 correct 49
Epoch  180  loss  0.63429055068607 correct 48
Epoch  190  loss  0.5689046942785623 correct 49
Epoch  200  loss  0.4100358220644913 correct 49
Epoch  210  loss  0.6853967635505138 correct 50
Epoch  220  loss  0.1918442007926213 correct 49
Epoch  230  loss  1.018953164299619 correct 49
Epoch  240  loss  0.19665474733505864 correct 49
Epoch  250  loss  0.07410798404321876 correct 49
Epoch  260  loss  0.9513855775022755 correct 50
Epoch  270  loss  0.1760497017705287 correct 50
Epoch  280  loss  0.22385015618495985 correct 50
Epoch  290  loss  0.3310768013167672 correct 50
Epoch  300  loss  0.6610841911289381 correct 49
Epoch  310  loss  0.5644459981094182 correct 49
Epoch  320  loss  0.46624568438225117 correct 50
Epoch  330  loss  0.633694275314323 correct 50
Epoch  340  loss  0.41948369707007727 correct 50
Epoch  350  loss  1.1632365102213993 correct 50
Epoch  360  loss  1.2076468997386813 correct 50
Epoch  370  loss  0.40945100960906833 correct 49
Epoch  380  loss  0.15956363712734556 correct 50
Epoch  390  loss  0.8457276229657693 correct 50
Epoch  400  loss  0.8499116055399694 correct 50
Epoch  410  loss  0.4548360396359752 correct 50
Epoch  420  loss  0.04546219821896195 correct 49
Epoch  430  loss  0.3901360275789234 correct 50
Epoch  440  loss  0.20356458873268457 correct 50
Epoch  450  loss  0.16519253554504892 correct 50
Epoch  460  loss  0.25686507311217105 correct 50
Epoch  470  loss  0.16157451191916725 correct 50
Epoch  480  loss  1.0846761092590766 correct 50
Epoch  490  loss  0.31076642740686866 correct 50

## Split dataset, CPU, Hidden 200, Rate 0.05
Epoch  0  loss  6.954855283140967 correct 36
Epoch  10  loss  6.757676509137842 correct 28
Epoch  20  loss  2.5378997143404693 correct 49
Epoch  30  loss  2.244025816250019 correct 48
Epoch  40  loss  1.437379113462034 correct 48
Epoch  50  loss  3.09734445856035 correct 47
Epoch  60  loss  2.6265600251983336 correct 48
Epoch  70  loss  2.101068890682968 correct 48
Epoch  80  loss  3.218717508716767 correct 48
Epoch  90  loss  1.8129326841745232 correct 48
Epoch  100  loss  1.0901575086988704 correct 48
Epoch  110  loss  1.308797868686105 correct 48
Epoch  120  loss  1.6374387325882611 correct 49
Epoch  130  loss  0.6534428842607976 correct 50
Epoch  140  loss  0.4086668839825767 correct 50
Epoch  150  loss  2.0887023997169507 correct 49
Epoch  160  loss  1.3885379474232993 correct 49
Epoch  170  loss  0.6270341499073667 correct 49
Epoch  180  loss  0.36325543452092923 correct 50
Epoch  190  loss  1.5751823067686164 correct 48
Epoch  200  loss  1.3805133229402458 correct 49
Epoch  210  loss  0.7628297186069618 correct 49
Epoch  220  loss  0.22472750671222816 correct 50
Epoch  230  loss  0.23197320471967417 correct 50
Epoch  240  loss  0.7452022344760482 correct 50
Epoch  250  loss  0.1241807998451125 correct 50
Epoch  260  loss  1.0610686119418693 correct 49
Epoch  270  loss  0.999888283588447 correct 50
Epoch  280  loss  0.7880267126533966 correct 50
Epoch  290  loss  0.4704132568320706 correct 50
Epoch  300  loss  0.3741375475066214 correct 50
Epoch  310  loss  0.3870271398796035 correct 50
Epoch  320  loss  0.7557468417303814 correct 50
Epoch  330  loss  0.6096828708519914 correct 50
Epoch  340  loss  0.9158390413455703 correct 50
Epoch  350  loss  0.5226108421709397 correct 50
Epoch  360  loss  0.24070842984212473 correct 50
Epoch  370  loss  0.10389091315066055 correct 50
Epoch  380  loss  0.43749728832702955 correct 50
Epoch  390  loss  0.30093802755594967 correct 50
Epoch  400  loss  0.4143179607901088 correct 50
Epoch  410  loss  0.5288420784201657 correct 50
Epoch  420  loss  0.24190106691483804 correct 50
Epoch  430  loss  0.5055340560180063 correct 50
Epoch  440  loss  0.5224573190271847 correct 50
Epoch  450  loss  0.006731915063455455 correct 50
Epoch  460  loss  0.05646488256678236 correct 50
Epoch  470  loss  0.30440980811804463 correct 50
Epoch  480  loss  0.7344097132120432 correct 50
Epoch  490  loss  0.13656682375309018 correct 50



