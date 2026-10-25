"""Read native sharded tensor storage without starting a distributed group."""
import pickle
import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

class SavedShardedTensor:
    """Deserialize saved local shards without creating a distributed group."""
    def __setstate__(self, state):
        self.saved_state = state

class SavedShardUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.distributed._shard.sharded_tensor.api' and name == 'ShardedTensor':
            return SavedShardedTensor
        return super().find_class(module, name)

class SavedShardPickle:
    Unpickler = SavedShardUnpickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)

def local_entries(value):
    if isinstance(value, DTensor):
        shape, offsets = compute_local_shape_and_global_offset(value.shape, value.device_mesh, value.placements)
        assert tuple(shape) == tuple(value._local_tensor.shape)
        return [(value._local_tensor, offsets)]
    if isinstance(value, SavedShardedTensor):
        return [(shard.tensor, shard.metadata.shard_offsets) for shard in value.saved_state[0]]
    if isinstance(value, torch.Tensor):
        return [(value, [0]*value.ndim)]
    raise TypeError(type(value))
