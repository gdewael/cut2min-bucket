from torch.utils.data import *
import math
import torch


def _partitions_to_len(n_samples, n_partitions, batch_size):
        # Count the number of samples per partition
        samples_per_partition = [
            math.ceil(n_samples / n_partitions)
        ] * n_partitions

        # The last partition may have fewer samples
        samples_per_partition[-1] -= (n_samples // n_partitions) % n_partitions

        # Count the number of batches per partition and sum
        len_ = sum([math.ceil(samples / batch_size) for samples in samples_per_partition])
        return len_


class BucketBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset,
        seqlens, # torch.Tensor (n, )
        batch_size,
        n_partitions=100,
        indices=None, # None or list
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
            RandomSampler(indices),
            math.ceil(len_dataset / n_partitions),
            False
        )
        self.indices = indices

        self._len = _partitions_to_len(len_dataset, n_partitions, batch_size)

    def __iter__(self):
        # For every partition, order all indices in it by seq. len
        indices_per_partition_ordered = []
        for partition in self.partitioner:
            partition_indices = self.indices[partition]

            partition_asort_seqlens = torch.argsort(self.seqlens[partition], descending=True)
            partition_indices_in_order = list(partition_indices[partition_asort_seqlens.numpy()])
            indices_per_partition_ordered.append(partition_indices_in_order)

        # Then iterate through all partitions
        for partition_indices in indices_per_partition_ordered:
            # Make batches per partition, then randomly shuffle around
            # The shuffling prevents that the smallest batches will always be first
            for batch in SubsetRandomSampler(list(BatchSampler(partition_indices, self.batch_size, self.drop_last))):
                yield batch

    def __len__(self):
        return self._len
    

class DistributedBucketSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        n_partitions = 100,
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
            drop_last=drop_last
        )

        self.batch_size = batch_size
        self.n_partitions = n_partitions

        self._len = _partitions_to_len(self.num_samples, n_partitions, batch_size)
    def __iter__(self):
        # Inherit a list of indices from parent class DistributedSampler
        indices = list(super().__iter__())

        # Use it to create a bucketbatchSampler
        batch_sampler = BucketBatchSampler(
            self.dataset,
            batch_size=self.batch_size,
            n_partitions=self.n_partitions,
            indices = indices
            )
        return iter(batch_sampler)

    def __len__(self):
        return self._len

