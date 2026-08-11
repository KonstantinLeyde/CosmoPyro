Drawing Random Fields
=====================

The ``RealField`` class can be used standalone to generate and visualise
random draws from a Gaussian Random Field (GRF) with any power spectrum.


1-D field
---------
The following compares draws from a 1-D GRF, for two boxes of same size but different resolution. 

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import matplotlib.pyplot as plt
   from cosmopyro.field_utils import field

   box_range = jnp.array([[0.0, 1.0]])
   box_shape_ds = [[50], [50000]]

   def my_power_spectrum(k):
       return 400 / (20 + (k ** 2) ** 2)

   fig, ax = plt.subplots()
   for box_shape_d in box_shape_ds:
       f = field.RealField(
           box_range_d=box_range,
           box_shape_d=box_shape_d,
           power_spectrum_of_k=my_power_spectrum,
           replace_FT_with_packing=False,
       )
       f.sample_gaussian_F_whitened_fourier(jax.random.PRNGKey(2))
       f.gaussian_F_whitened_fourier = f.gaussian_F_whitened_fourier.at[0].set(0.0)
       f.compute_gaussian_F_spatial_from_gaussian_F_whitened({})

       x = jnp.linspace(0, 1, f.box_shape_d[0])
       ax.plot(x, jnp.exp(f.gaussian_F_spatial), label=f"N = {box_shape_d[0]}")

   ax.set_xlabel("x")
   ax.set_ylabel("exp(field)")
   ax.legend()
   plt.show()


2-D field
---------

The following produces one random draw from a 2-D GRF. 
You can vary the power spectrum to see how it affects the structure of the field.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import matplotlib.pyplot as plt
   from cosmopyro.field_utils import field

   box_range = jnp.array([[0.0, 1.0], [0.0, 1.0]])
   box_shape_d = [150, 150]

   def my_power_spectrum(k1, k2):
       k = jnp.sqrt(k1 ** 2 + k2 ** 2)
       return 400 / (20 + (k ** 2) ** 2)

   f = field.RealField(
       box_range_d=box_range,
       box_shape_d=box_shape_d,
       power_spectrum_of_k=my_power_spectrum,
       replace_FT_with_packing=True,
   )
   f.sample_gaussian_F_whitened_fourier(jax.random.PRNGKey(2))
   f.gaussian_F_whitened_fourier = f.gaussian_F_whitened_fourier.at[0].set(0.0)
   f.compute_gaussian_F_spatial_from_gaussian_F_whitened({})

   plt.imshow(f.gaussian_F_spatial, origin="lower", aspect="auto")
   plt.colorbar(label="field value")
   plt.xlabel("axis 1")
   plt.ylabel("axis 0")
   plt.show()
