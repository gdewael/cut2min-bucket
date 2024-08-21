<div align="center">
<h1>cut2min-bucket</h1>

A PyTorch Batch Sampler that buckets by input length and cuts to min size in batch

[![PyPi Version](https://img.shields.io/pypi/v/cut2min-bucket.svg)](https://pypi.python.org/pypi/cut2min-bucket/)
[![GitHub license](https://img.shields.io/github/license/gdewael/cut2min-bucket)](https://github.com/gdewael/cut2min-bucket/blob/main/LICENSE)

</div>

This package provides 2 utilities:
1. `cut2min_bucket.DatasetWrapper` to eliminate padding and cut to min size in batch
2. `cut2min_bucket.BucketBatchSampler` a batch sampler that buckets by input length.

In addition, we provide a Distributed Data Parallel version of the batch sampler: `cut2min_bucket.DistributedBucketBatchSampler`.

A detailed motivation for this package can be found on [my blog](https://gdewael.github.io/blog/flashattnvarlen/).


Simple example:
```python
import cut2min_bucket
import torch
import numpy as np

X = []
for _ in range(10000):
    X.append(torch.tensor(np.random.randn(torch.randint(size=(), low=2, high=1000),)))

seqlens = torch.tensor([len(x) for x in X])

X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
y = (torch.rand(10000)>0.5).int()

dataset = torch.utils.data.TensorDataset(X, y)

dataset = cut2min_bucket.DatasetWrapper(
    dataset, seqlens,
    index_or_key=0
)

batch_sampler = cut2min_bucket.BucketBatchSampler(
    dataset,
    seqlens,
    batch_size=8,
    n_partitions=5
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_sampler=batch_sampler,
    collate_fn=dataset.collate_fn,
)

next(iter(dataloader))
```