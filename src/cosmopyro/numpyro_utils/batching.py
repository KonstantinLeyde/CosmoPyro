import jax
import jax.numpy as jnp

__all__ = [
    "apply_batched_operation_1d",
    "get_safe_shape",
    "get_safe_size",
]


def get_safe_size(x):
    return x.size if hasattr(x, "size") else 1


def get_safe_shape(x):
    return x.shape if hasattr(x, "shape") else ()


# --- 2. The Generic Batching Engine ---
def apply_batched_operation_1d(
    data_1d, operation_fn, batch_size, checkpoint_gradients=True
):
    """
    Takes a PyTree of 1D arrays, batches them, applies 'operation_fn' using jax.lax.scan,
    and returns a single concatenated 1D result.

    Args:
        data_1d: PyTree where leaves are 1D arrays of shape (N_total,)
        operation_fn: Function that takes (data_batch) and returns (result_batch)
        batch_size: Integer size of chunks
        checkpoint_gradients: Rematerialize each batch in the backward pass.
            Without this, reverse-mode AD stores the intermediates of every
            batch and peak memory grows with N_total despite the batching;
            with it, peak memory scales with batch_size at the cost of one
            extra forward pass per batch.
    """
    if checkpoint_gradients:
        operation_fn = jax.checkpoint(operation_fn)

    # Identify total size from the first leaf
    leaves = jax.tree_util.tree_leaves(data_1d)
    total_elements = leaves[0].shape[0]

    num_batches = total_elements // batch_size
    cutoff = num_batches * batch_size

    if num_batches == 0:
        return operation_fn(data_1d)

    # A. Prepare Batched Input (num_batches, batch_size)
    def has_non_scale_shape(x):
        # 1. Must have a .shape attribute
        # 2. Must have dimensions (len(shape) > 0). Scalars shape is ()
        # 3. First dimension must match the total data size
        return hasattr(x, "shape") and len(x.shape) > 0 and x.shape[0] == total_elements

    # C. Prepare Batches for Scan
    def reshape_for_scan(x):
        if has_non_scale_shape(x):
            # It's a large array -> Reshape to (num_batches, batch_size)
            return jnp.reshape(x[:cutoff], (num_batches, batch_size))
        else:
            # It's a scalar/int -> Broadcast to (num_batches, ...)
            # This makes 'scan' pass the same value to every iteration
            return jnp.broadcast_to(x, (num_batches,))

    batched_input = jax.tree_util.tree_map(reshape_for_scan, data_1d)

    # B. Define Scan Wrapper
    def scan_body(carry, batch):
        # Apply the user's operation
        result = operation_fn(batch)
        # Carry is unused (None), result is stacked
        return None, result

    # C. Run Scan
    # scanned_results shape: (num_batches, batch_size)
    _, scanned_results = jax.lax.scan(scan_body, None, batched_input)

    # Flatten back to 1D
    processed_main = jnp.reshape(scanned_results, (cutoff,))

    # D. Handle Remainder
    if total_elements > cutoff:

        def slice_remainder(x):
            # Only slice the large data arrays
            return x[cutoff:] if has_non_scale_shape(x) else x

        remainder_input = jax.tree_util.tree_map(slice_remainder, data_1d)
        remainder_result = operation_fn(remainder_input)
        return jnp.concatenate([processed_main, remainder_result], axis=0)

    return processed_main
