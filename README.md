### Alpha-ReLU

This package provides a pytorch implementation of α-ReLU and α-ReLU loss for α=1.5 

*Requirements:* python 3, pytorch >= 1.0

The library based on the implementation of [entmax](https://github.com/deep-spin/entmax)

## Example

```python
In [1]: import torch

In [2]: from torch.nn.functional import softmax

In [2]: from alpharelu import relu15

In [4]: x = torch.tensor([-2, 0.3, 0.5])

In [5]: softmax(x, dim=0)
Out[5]: tensor([0.0432, 0.4307, 0.5261])

In [6]: relu15(x, dim=0)
Out[6]: tensor([0.0000, 0.0225, 0.0625])
```

Loss
```python
In [1]: import torch

In [2]: from alpharelu import ReLU15Loss
```
Note, that *relu15* and *ReLU15Loss* takes optional **tau** argument (default: 0.0) which is responsible to the value of <img src="https://render.githubusercontent.com/render/math?math=\tau"> in ![image](https://user-images.githubusercontent.com/29367747/167902137-8c89008b-e879-444f-aa08-217e00a848ff.png)
Optimal value of <img src="https://render.githubusercontent.com/render/math?math=\tau"> can be obtained by:
```python
from alpharelu import get_tau
optimal_tau = get_tau(x/2)
relu15(x, dim=-1, tau=optimal_tau)
```
where **x** is the logits of the first batch. Note, that for 1.5-ReLU, you need to pass **x/2** to *get_tau* function.

## Installation
```
pip install alpharelu
```
## Citations

[Speeding Up Entmax](https://arxiv.org/abs/2111.06832)

```
@article{DBLP:journals/corr/abs-2111-06832,
  author    = {Maxat Tezekbayev and
               Vassilina Nikoulina and
               Matthias Gall{\'{e}} and
               Zhenisbek Assylbekov},
  title     = {Speeding Up Entmax},
  journal   = {CoRR},
  volume    = {abs/2111.06832},
  year      = {2021},
  url       = {https://arxiv.org/abs/2111.06832},
  eprinttype = {arXiv},
  eprint    = {2111.06832},
  timestamp = {Tue, 16 Nov 2021 12:12:31 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2111-06832.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
