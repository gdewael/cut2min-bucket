from torch.utils.data import *
import math
import numpy as np
import torch

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, seqlens, index_or_key = None):
        super().__init__()
        self.dataset = dataset
        self.seqlens = seqlens
        self.ix_or_key = index_or_key

    def __getitem__(self, index):
        sample = self.dataset[index]
        seqlen = self.seqlens[index]

        return self._eliminate_padding(sample, seqlen, self.ix_or_key)

    def __len__(self):
        return len(self.dataset)
    

    def collate_fn(self, batch):
        first_sample = batch[0]
        if isinstance(first_sample, tuple):
            batch = [list(sample) for sample in batch]
            first_sample = batch[0]
        

        if isinstance(first_sample, list):
            error_str = "If sample is a list of objects, `index_or_key` should be an integer or list of ints indexing into that list."
            if isinstance(self.ix_or_key, int):
                try: first_sample[self.ix_or_key]
                except: raise ValueError(error_str)
                
                tocut = [sample.pop(self.ix_or_key) for sample in batch]
                collated_batch = default_collate(batch)
                collated_batch.insert(self.ix_or_key, default_collate(self._cut_to_uniform_size(tocut)))

            elif isinstance(self.ix_or_key, list):
                for padded_ixes in sorted(self.ix_or_key):
                    try: first_sample[padded_ixes]
                    except: raise ValueError(error_str)

                tocuts = [
                    [sample.pop(padded_ixes-ix_in_) for sample in batch]
                    for ix_in_, padded_ixes in enumerate(sorted(self.ix_or_key))
                ]

                collated_batch = default_collate(batch)
                for padded_ixes, tocut in zip(sorted(self.ix_or_key), tocuts):
                    collated_batch.insert(padded_ixes, default_collate(self._cut_to_uniform_size(tocut)))

            return collated_batch

        elif isinstance(first_sample, dict):
            error_str = "If sample is a dict of objects, `index_or_key` should be a key or list of keys in that dict"
            if isinstance(self.ix_or_key, (int, float, str, tuple)):
                try: first_sample[self.ix_or_key]
                except: raise ValueError(error_str)
                
                tocut = [sample.pop(self.ix_or_key) for sample in batch]
                collated_batch = default_collate(batch)
                collated_batch[self.ix_or_key] = default_collate(self._cut_to_uniform_size(tocut))

            elif isinstance(self.ix_or_key, list):
                for padded_keys in sorted(self.ix_or_key):
                    try: first_sample[padded_keys]
                    except: raise ValueError(error_str)
                tocuts = [
                    [sample.pop(padded_keys) for sample in batch]
                    for padded_keys in sorted(self.ix_or_key)
                ]
                collated_batch = default_collate(batch)
                for padded_keys, tocut in zip(sorted(self.ix_or_key), tocuts):
                    collated_batch[padded_keys] = default_collate(self._cut_to_uniform_size(tocut))
                    
            return collated_batch
        else:
            if self.ix_or_key is not None:
                raise ValueError("If sample is not a list or dict of objects, cannot specify an index_or_key")
            
            return default_collate(self._cut_to_uniform_size(batch))         

    @staticmethod
    def _cut_to_uniform_size(list_of_objects):
        min_len = min([b.shape[-1] for b in list_of_objects])
        return [b[..., :min_len] for b in list_of_objects]
    
    @staticmethod
    def _eliminate_padding(sample, seqlen, padded_ix_or_key):
        if isinstance(sample, tuple):
            sample = list(sample)

        if isinstance(sample, list):
            error_str = "If sample is a list of objects, `index_or_key` should be an integer or list of ints indexing into that list."
            if isinstance(padded_ix_or_key, int):
                try: sample[padded_ix_or_key]
                except: raise ValueError(error_str)
                
                tocut = sample.pop(padded_ix_or_key)
                sample.insert(padded_ix_or_key, tocut[..., :seqlen])
            elif isinstance(padded_ix_or_key, list):
                for padded_ixes in sorted(padded_ix_or_key):
                    try: sample[padded_ixes]
                    except: raise ValueError(error_str)
                    
                    tocut = sample.pop(padded_ixes)
                    sample.insert(padded_ixes, tocut[..., :seqlen])
            else:
                raise ValueError(error_str)

            return sample
        
        elif isinstance(sample, dict):
            error_str = "If sample is a dict of objects, `index_or_key` should be a key or list of keys in that dict"
            if isinstance(padded_ix_or_key, (int, float, str, tuple)):
                try: sample[padded_ix_or_key]
                except: raise ValueError(error_str)
                
                tocut = sample.pop(padded_ix_or_key)
                sample[padded_ix_or_key] = tocut[..., :seqlen]
            elif isinstance(padded_ix_or_key, list):
                for padded_keys in padded_ix_or_key:
                    try: sample[padded_keys]
                    except: raise ValueError(error_str)
                    
                    tocut = sample.pop(padded_keys)
                    sample[padded_keys] = tocut[..., :seqlen]

            return sample
        
        else:
            if padded_ix_or_key is not None:
                raise ValueError("If batch is not a list or dict of objects, cannot specify an index_or_key")
            return sample[..., :seqlen]