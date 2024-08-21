from torch.utils.data import *
import math
import torch


def _partitions_to_len(n_samples, n_partitions, batch_size):
    # Count the number of samples per partition
    samples_per_partition = [math.ceil(n_samples / n_partitions)] * n_partitions

    # The last partition may have fewer samples
    samples_per_partition[-1] -= (n_samples // n_partitions) % n_partitions

    # Count the number of batches per partition and sum
    len_ = sum([math.ceil(samples / batch_size) for samples in samples_per_partition])
    return len_


class BucketBatchSampler(BatchSampler):
    """BucketBatchSampler.
    See https://gdewael.github.io/blog/flashattnvarlen/ for explanation.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset or any compatible object.
        PyTorch Dataset. Either a torch.utils.data.Dataset itself or any object that has __len__ or __getitem__ implemented.
    seqlens : list, 1D np.ndarray, or 1D torch.tensor
        All sequence lengths as integers. Should have the same length as `dataset`.
    batch_size: int.
        Mini batch size
    n_partitions: int, optional
        Before sorting samples by size, partition dataset into this many subsets.
        Increase the number of partitions to incease stochasticity of data sampling.
        If put to 1, samples will be grouped together the same way each epoch.
        Default = 100.
    indices: list, optional
        Used in Distributed version. Subsets the provided dataset
        Default = None.
    drop_last: boolean, optional
        In every partition, whether to drop the last batch which could not be fully "filled" to the batch size. Default = False.
    """
    def __init__(
        self,
        dataset,
        seqlens,  # torch.Tensor (n, )
        batch_size,
        n_partitions=100,
        indices=None,  # None or list
        drop_last=False,
    ):
        super().__init__(dataset, batch_size, drop_last)

        # `indices` subsamples the dataset in the case of a Distributed Data setting
        if indices is not None:
            len_dataset = len(indices)
            self.seqlens = seqlens[indices]
            indices = torch.tensor(indices)
        else:
            len_dataset = len(dataset)
            self.seqlens = seqlens
            indices = torch.arange(len_dataset)

        # randomly partition dataset in n_partitions
        self.partitioner = BatchSampler(
            RandomSampler(indices), math.ceil(len_dataset / n_partitions), False
        )
        self.indices = indices

        self._len = _partitions_to_len(len_dataset, n_partitions, batch_size)

    def __iter__(self):
        # For every partition, order all indices in it by seq. len
        indices_per_partition_ordered = []
        for partition in self.partitioner:
            partition_indices = self.indices[partition]

            partition_asort_seqlens = torch.argsort(
                self.seqlens[partition], descending=True
            )
            partition_indices_in_order = list(
                partition_indices[partition_asort_seqlens.numpy()]
            )
            indices_per_partition_ordered.append(partition_indices_in_order)

        # Then iterate through all partitions
        for partition_indices in indices_per_partition_ordered:
            # Make batches per partition, then randomly shuffle around
            # The shuffling prevents that the smallest batches will always be first
            for batch in SubsetRandomSampler(
                list(BatchSampler(partition_indices, self.batch_size, self.drop_last))
            ):
                yield batch

    def __len__(self):
        return self._len


class DistributedBucketBatchSampler(DistributedSampler):
    """DistributedBucketBatchSampler.
    See https://gdewael.github.io/blog/flashattnvarlen/ for explanation.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset or any compatible object.
        PyTorch Dataset. Either a torch.utils.data.Dataset itself or any object that has __len__ or __getitem__ implemented.
    seqlens : list, 1D np.ndarray, or 1D torch.tensor
        All sequence lengths as integers. Should have the same length as `dataset`.
    batch_size: int.
        Mini batch size
    n_partitions: int, optional
        Before sorting samples by size, partition dataset into this many subsets.
        Increase the number of partitions to incease stochasticity of data sampling.
        If put to 1, samples will be grouped together the same way each epoch.
    num_replicas: int, optional
        see https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    shuffle: boolean, optional
        see https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    seed: int, optional
        see https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    drop_last: boolean, optional
        see https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
    """
    def __init__(
        self,
        dataset,
        seqlens,
        batch_size,
        n_partitions=100,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

        self.batch_size = batch_size
        self.n_partitions = n_partitions
        self.seqlens = seqlens
        self.drop_last = drop_last

        self._len = _partitions_to_len(self.num_samples, n_partitions, batch_size)

    def __iter__(self):
        # Inherit a list of indices from parent class DistributedSampler
        indices = list(super().__iter__())

        # Use it to create a bucketbatchSampler
        batch_sampler = BucketBatchSampler(
            self.dataset,
            self.seqlens,
            self.batch_size,
            n_partitions=self.n_partitions,
            indices=indices,
            drop_last=self.drop_last,
        )
        return iter(batch_sampler)

    def __len__(self):
        return self._len
