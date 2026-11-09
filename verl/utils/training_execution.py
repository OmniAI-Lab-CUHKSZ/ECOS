"""Opt-in execution changes; preserve loss units and optimizer boundaries."""
from contextlib import contextmanager, nullcontext


@contextmanager
def accumulation_microbatches(module, batches, enabled=False):
    """Delay synchronization until the last microbatch, never the next step.

    A managed generator closes no_sync even if the caller raises or breaks.
    Existing losses, scale factors, masks and clipping are unchanged.
    """
    def iterate():
        for index, batch in enumerate(batches):
            context = (module.no_sync() if enabled and index + 1 < len(batches)
                       else nullcontext())
            with context:
                yield batch
    iterator = iterate()
    try:
        yield iterator
    finally:
        iterator.close()
