from functools import partial

import jax
import jax.numpy as jnp

__all__ = [
    "even_pack_2d",
    "even_pack_3d",
    "even_unpack_2d",
    "even_unpack_3d",
    "odd_pack_2d",
    "odd_unpack_2d",
    "pack_fft_values_2d",
    "pack_fft_values_3d",
    "unpack_fft_values_2d",
    "unpack_fft_values_3d",
]


@partial(jax.jit, static_argnums=(1, 2, 3))
def even_pack_3d(values, Nx, Ny, Nz):
    """Pack the value for an even-by-even-by-even FFT"""
    n1x = Nx // 2 + 1
    n1y = Ny // 2 + 1
    n1z = Nz // 2 + 1

    # real part

    # first slice

    thin_real = jax.lax.dynamic_slice(values, (0, 1, 0), (Nx, n1y - 2, 1))

    first_real_slice = jax.lax.dynamic_slice(values, (1, 0, 0), (n1x - 2, 1, 1))
    first_real = jnp.vstack(
        [
            jnp.sqrt(2) * jax.lax.dynamic_slice(values, (0, 0, 0), (1, 1, 1)),
            first_real_slice,
            jnp.sqrt(2) * jax.lax.dynamic_slice(values, (n1x - 1, 0, 0), (1, 1, 1)),
            jnp.flip(first_real_slice, axis=0),
        ]
    )

    bla = jax.lax.dynamic_slice(thin_real, (0, 0, 0), (1, n1y - 2, 1))
    bla_rot = jax.lax.dynamic_slice(thin_real, (1, 0, 0), (Nx - 1, n1y - 2, 1))
    bla_rott = jnp.flip(bla_rot, axis=(0, 1))

    bla_flipped = jnp.flip(bla, axis=1)
    thin_real_rot = jnp.concatenate([bla_flipped, bla_rott], axis=0)

    middle_real_slice = jax.lax.dynamic_slice(values, (1, n1y - 1, 0), (n1x - 2, 1, 1))
    middle_real = jnp.vstack(
        [
            jnp.sqrt(2) * jax.lax.dynamic_slice(values, (0, n1y - 1, 0), (1, 1, 1)),
            middle_real_slice,
            jnp.sqrt(2)
            * jax.lax.dynamic_slice(values, (n1x - 1, n1y - 1, 0), (1, 1, 1)),
            jnp.flip(middle_real_slice, axis=0),
        ]
    )

    first_slice_full = jnp.concatenate(
        [first_real, thin_real, middle_real, thin_real_rot], axis=1
    )

    # last slice
    thin_real = jax.lax.dynamic_slice(values, (0, 1, n1z - 1), (Nx, n1y - 2, 1))

    first_real_slice = jax.lax.dynamic_slice(values, (1, 0, n1z - 1), (n1x - 2, 1, 1))
    first_real = jnp.vstack(
        [
            jnp.sqrt(2) * jax.lax.dynamic_slice(values, (0, 0, n1z - 1), (1, 1, 1)),
            first_real_slice,
            jnp.sqrt(2)
            * jax.lax.dynamic_slice(values, (n1x - 1, 0, n1z - 1), (1, 1, 1)),
            jnp.flip(first_real_slice, axis=0),
        ]
    )

    bla = jax.lax.dynamic_slice(thin_real, (0, 0, 0), (1, n1y - 2, 1))
    bla_rot = jax.lax.dynamic_slice(thin_real, (1, 0, 0), (Nx - 1, n1y - 2, 1))
    bla_rott = jnp.flip(bla_rot, axis=(0, 1))

    bla_flipped = jnp.flip(bla, axis=1)
    thin_real_rot = jnp.concatenate([bla_flipped, bla_rott], axis=0)

    middle_real_slice = jax.lax.dynamic_slice(
        values, (1, n1y - 1, n1z - 1), (n1x - 2, 1, 1)
    )
    middle_real = jnp.vstack(
        [
            jnp.sqrt(2)
            * jax.lax.dynamic_slice(values, (0, n1y - 1, n1z - 1), (1, 1, 1)),
            middle_real_slice,
            jnp.sqrt(2)
            * jax.lax.dynamic_slice(values, (n1x - 1, n1y - 1, n1z - 1), (1, 1, 1)),
            jnp.flip(middle_real_slice, axis=0),
        ]
    )

    last_slice_full = jnp.concatenate(
        [first_real, thin_real, middle_real, thin_real_rot], axis=1
    )

    boring = jax.lax.dynamic_slice(values, (0, 0, 1), (Nx, Ny, n1z - 2))
    fft_real = jnp.concatenate([first_slice_full, boring, last_slice_full], axis=2)

    # imaginary part

    # first slice
    thin_imag = jax.lax.dynamic_slice(values, (0, n1y, 0), (Nx, n1y - 2, 1))

    first_real_slice = jax.lax.dynamic_slice(values, (n1x, 0, 0), (n1x - 2, 1, 1))
    first_real = jnp.vstack(
        [
            jnp.zeros((1, 1, 1)),
            -jnp.flip(first_real_slice, axis=0),
            jnp.zeros((1, 1, 1)),
            first_real_slice,
        ]
    )

    bla = jax.lax.dynamic_slice(thin_imag, (0, 0, 0), (1, n1y - 2, 1))
    bla_rot = jax.lax.dynamic_slice(thin_imag, (1, 0, 0), (Nx - 1, n1y - 2, 1))
    bla_rott = jnp.flip(bla_rot, axis=(0, 1))

    bla_flipped = jnp.flip(bla, axis=1)
    thin_imag_rot = jnp.concatenate([bla_flipped, bla_rott], axis=0)

    middle_real_slice = jax.lax.dynamic_slice(
        values, (n1x, n1y - 1, 0), (n1x - 2, 1, 1)
    )
    middle_real = jnp.vstack(
        [
            jnp.zeros((1, 1, 1)),
            -jnp.flip(middle_real_slice, axis=0),
            jnp.zeros((1, 1, 1)),
            middle_real_slice,
        ]
    )

    first_slice_full = jnp.concatenate(
        [first_real, thin_imag, middle_real, -thin_imag_rot], axis=1
    )

    # last slice
    thin_imag = jax.lax.dynamic_slice(values, (0, n1y, n1z - 1), (Nx, n1y - 2, 1))

    first_real_slice = jax.lax.dynamic_slice(values, (n1x, 0, n1z - 1), (n1x - 2, 1, 1))
    first_real = jnp.vstack(
        [
            jnp.zeros((1, 1, 1)),
            -jnp.flip(first_real_slice, axis=0),
            jnp.zeros((1, 1, 1)),
            first_real_slice,
        ]
    )

    bla = jax.lax.dynamic_slice(thin_imag, (0, 0, 0), (1, n1y - 2, 1))
    bla_rot = jax.lax.dynamic_slice(thin_imag, (1, 0, 0), (Nx - 1, n1y - 2, 1))
    bla_rott = jnp.flip(bla_rot, axis=(0, 1))

    bla_flipped = jnp.flip(bla, axis=1)
    thin_imag_rot = jnp.concatenate([bla_flipped, bla_rott], axis=0)

    middle_real_slice = jax.lax.dynamic_slice(
        values, (n1x, n1y - 1, n1z - 1), (n1x - 2, 1, 1)
    )
    middle_real = jnp.vstack(
        [
            jnp.zeros((1, 1, 1)),
            -jnp.flip(middle_real_slice, axis=0),
            jnp.zeros((1, 1, 1)),
            middle_real_slice,
        ]
    )

    last_slice_full = jnp.concatenate(
        [first_real, thin_imag, middle_real, -thin_imag_rot], axis=1
    )

    boring = jnp.flip(
        jax.lax.dynamic_slice(values, (0, 0, n1z), (Nx, Ny, n1z - 2)), axis=2
    )
    fft_imag = jnp.concatenate([first_slice_full, boring, last_slice_full], axis=2)

    return fft_real + 1j * fft_imag


@partial(jax.jit, static_argnums=(1,))
def even_unpack_3d(values, N):
    raise NotImplementedError

    n1 = N // 2 + 1
    thin_slice = jax.lax.dynamic_slice(values, (0, 0, 1), (N, N, n1 - 2))
    first_slice = jax.lax.dynamic_slice(values, (0, 1, 0), (N, n1 - 2, 1))
    first = jnp.concatenate(
        [
            0.5 * values[:, 0, 0].real.reshape(N, 1, 1),
            first_slice.real,
            0.5 * values[:, n1 - 1, 0].real.reshape(N, 1, 1),
            -jnp.flip(first_slice.imag, axis=1),
        ],
        axis=1,
    )

    last_slice = jax.lax.dynamic_slice(values, (0, 1, n1 - 1), (N, n1 - 2, 1))
    last = jnp.concatenate(
        [
            0.5 * values[:, 0, n1 - 1].real.reshape(N, 1, 1),
            last_slice.real,
            0.5 * values[:, n1 - 1, n1 - 1].real.reshape(N, 1, 1),
            -jnp.flip(last_slice.imag, axis=1),
        ],
        axis=1,
    )

    delta = thin_slice.shape[1] - first.shape[1]
    first = jnp.pad(first, ((0, 0), (0, delta), (0, 0)))
    last = jnp.pad(last, ((0, 0), (0, delta), (0, 0)))

    out = jnp.concatenate(
        [first, thin_slice.real, last, jnp.flip(thin_slice.imag, axis=2)], axis=2
    )

    delta = N - out.shape[2]
    return jnp.pad(out, ((0, 0), (0, 0), (0, delta)))


@jax.jit
def pack_fft_values_3d(values):
    """
    Take values in a NxNxN array and re-arange them into
    the shape of the output of rFFT on an array of the
    same size by adding in the correct symmetries that
    come from a real image input.

    Doing this on a grid of values drawn from a standard
    Normal will result in the rFFT of a white noise image.

    This is used for drawing a white noise FFT in Fourier
    space so it can be "colored" by a power spectrum
    directly.

    parameters
    ----------
    values: The values to be packed, must be a cubic array
    """
    Nx, Ny, Nz = values.shape
    assert Nx == Ny, "First two slices must be equal"
    assert Nz % 2 == 0, "Last slice must have even size"

    even_pack_n = partial(even_pack_3d, Nx=Nx, Ny=Ny, Nz=Nz)
    odd_pack_n = partial(odd_pack_3d, Nx=Nx, Ny=Ny, Nz=Nz)
    return jax.lax.cond(Nx % 2 == 0, even_pack_n, odd_pack_n, jnp.sqrt(0.5) * values)


vpack_fft_values_3d = jax.vmap(pack_fft_values_3d)


@jax.jit
def unpack_fft_values_3d(values):
    Nz, _, _ = values.shape
    even_unpack_n = partial(even_unpack_3d, N=Nz)
    odd_unpack_n = partial(odd_unpack_3d, N=Nz)
    return jax.lax.cond(
        Nz % 2 == 0, even_unpack_n, odd_unpack_n, jnp.sqrt(2.0) * values
    )


vunpack_fft_values_3d = jax.vmap(unpack_fft_values_3d)


# from Coleman
@partial(jax.jit, static_argnums=(1,))
def odd_pack_2d(values, N):
    """Pack the value for an odd-by-odd FFT"""
    n1 = N // 2 + 1
    thin_real = jax.lax.dynamic_slice(values, (0, 1), (N, n1 - 1))
    thin_imag = jnp.flip(jax.lax.dynamic_slice(values, (0, n1), (N, n1 - 1)), axis=1)

    first_real_slice = jax.lax.dynamic_slice(values, (1, 0), (n1 - 1, 1))
    first_real = jnp.vstack(
        [
            2 * values[0, 0].reshape(1, 1),
            first_real_slice,
            jnp.flip(first_real_slice, axis=0),
        ]
    )

    first_imag_slice = jax.lax.dynamic_slice(values, (n1, 0), (n1 - 1, 1))
    first_imag = jnp.vstack(
        [
            jnp.zeros((1, 1)),
            -jnp.flip(first_imag_slice, axis=0),
            first_imag_slice,
        ]
    )

    fft_real = jnp.hstack([first_real[: thin_real.shape[0]], thin_real])
    fft_imag = jnp.hstack([first_imag[: thin_imag.shape[0]], thin_imag])
    return fft_real + 1j * fft_imag


@partial(jax.jit, static_argnums=(1,))
def odd_unpack_2d(values, N):
    n1 = N // 2 + 1
    thin_slice = jax.lax.dynamic_slice(values, (0, 1), (N, n1 - 1))
    first_slice = jax.lax.dynamic_slice(values, (1, 0), (n1 - 1, 1))
    first = jnp.vstack(
        [
            0.5 * values[0, 0].real.reshape(1, 1),
            first_slice.real,
            -jnp.flip(first_slice.imag, axis=0),
        ]
    )

    delta = first.shape[0] - thin_slice.shape[0]
    thin_slice = jnp.pad(thin_slice, ((0, delta), (0, 0)))

    return jax.lax.dynamic_slice(
        jnp.hstack([first, thin_slice.real, jnp.flip(thin_slice.imag, axis=1)]),
        (0, 0),
        (N, N),
    )


@partial(jax.jit, static_argnums=(1,))
def even_pack_2d(values, N):
    """Pack the value for an even-by-even FFT"""
    n1 = N // 2 + 1
    thin_real = jax.lax.dynamic_slice(values, (0, 1), (N, n1 - 2))
    thin_imag = jnp.flip(jax.lax.dynamic_slice(values, (0, n1), (N, n1 - 2)), axis=1)

    first_real_slice = jax.lax.dynamic_slice(values, (1, 0), (n1 - 2, 1))
    first_real = jnp.vstack(
        [
            2 * jax.lax.dynamic_slice(values, (0, 0), (1, 1)),
            first_real_slice,
            2 * jax.lax.dynamic_slice(values, (n1 - 1, 0), (1, 1)),
            jnp.flip(first_real_slice, axis=0),
        ]
    )

    last_real_slice = jax.lax.dynamic_slice(values, (1, n1 - 1), (n1 - 2, 1))
    last_real = jnp.vstack(
        [
            2 * jax.lax.dynamic_slice(values, (0, n1 - 1), (1, 1)),
            last_real_slice,
            2 * jax.lax.dynamic_slice(values, (n1 - 1, n1 - 1), (1, 1)),
            jnp.flip(last_real_slice, axis=0),
        ]
    )

    first_imag_slice = jax.lax.dynamic_slice(values, (n1, 0), (n1 - 2, 1))
    first_imag = jnp.vstack(
        [
            jnp.zeros((1, 1)),
            -jnp.flip(first_imag_slice, axis=0),
            jnp.zeros((1, 1)),
            first_imag_slice,
        ]
    )

    last_imag_slice = jax.lax.dynamic_slice(values, (n1, n1 - 1), (n1 - 2, 1))
    last_imag = jnp.vstack(
        [
            jnp.zeros((1, 1)),
            -jnp.flip(last_imag_slice, axis=0),
            jnp.zeros((1, 1)),
            last_imag_slice,
        ]
    )

    delta = thin_real.shape[0] - first_real.shape[0]
    first_real = jnp.pad(first_real, ((0, delta), (0, 0)))
    last_real = jnp.pad(last_real, ((0, delta), (0, 0)))
    first_imag = jnp.pad(first_imag, ((0, delta), (0, 0)))
    last_imag = jnp.pad(last_imag, ((0, delta), (0, 0)))

    fft_real = jnp.hstack([first_real, thin_real, last_real])
    fft_imag = jnp.hstack([first_imag, thin_imag, last_imag])
    return fft_real + 1j * fft_imag


@partial(jax.jit, static_argnums=(1,))
def even_unpack_2d(values, N):
    n1 = N // 2 + 1
    thin_slice = jax.lax.dynamic_slice(values, (0, 1), (N, n1 - 2))
    first_slice = jax.lax.dynamic_slice(values, (1, 0), (n1 - 2, 1))
    first = jnp.vstack(
        [
            0.5 * values[0, 0].real.reshape(1, 1),
            first_slice.real,
            0.5 * values[n1 - 1, 0].real.reshape(1, 1),
            -jnp.flip(first_slice.imag, axis=0),
        ]
    )
    last_slice = jax.lax.dynamic_slice(values, (1, n1 - 1), (n1 - 2, 1))
    last = jnp.vstack(
        [
            0.5 * values[0, n1 - 1].real.reshape(1, 1),
            last_slice.real,
            0.5 * values[n1 - 1, n1 - 1].real.reshape(1, 1),
            -jnp.flip(last_slice.imag, axis=0),
        ]
    )

    delta = thin_slice.shape[0] - first.shape[0]
    first = jnp.pad(first, ((0, delta), (0, 0)))
    last = jnp.pad(last, ((0, delta), (0, 0)))

    out = jnp.hstack([first, thin_slice.real, last, jnp.flip(thin_slice.imag, axis=1)])

    delta = N - out.shape[1]
    return jnp.pad(out, ((0, 0), (0, delta)))


@jax.jit
def pack_fft_values_2d(values):
    """
    Take values in a NxN array and re-arange them into
    the shape of the output of rFFT on an array of the
    same size by adding in the correct symmetries that
    come from a real image input.

    Doing this on a grid of values drawn from a standard
    Normal will result in the rFFT of a white noise image.

    This is used for drawing a white noise FFT in Fourier
    space so it can be "colored" by a power spectrum
    directly.

    parameters
    ----------
    values: The values to be packed, must be a square array
    """
    Ny, Nx = values.shape
    assert Ny == Nx, "Input array must be square"
    even_pack_n = partial(even_pack_2d, N=Nx)
    odd_pack_n = partial(odd_pack_2d, N=Nx)
    return jax.lax.cond(Nx % 2 == 0, even_pack_n, odd_pack_n, jnp.sqrt(0.5) * values)


vpack_fft_values_2d = jax.vmap(pack_fft_values_2d)


@jax.jit
def unpack_fft_values_2d(values):
    Ny, _ = values.shape
    even_unpack_n = partial(even_unpack_2d, N=Ny)
    odd_unpack_n = partial(odd_unpack_2d, N=Ny)
    return jax.lax.cond(
        Ny % 2 == 0, even_unpack_n, odd_unpack_n, jnp.sqrt(2.0) * values
    )


vunpack_fft_values_2d = jax.vmap(unpack_fft_values_2d)
