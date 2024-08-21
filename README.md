# cut2min-bucket

## Motivation and documentation WIP.

A PyTorch Batch Sampler that buckets by input length and cuts to min size in batch

This allows to use "vanilla" flash attention, which I have found to be vastly superior.


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