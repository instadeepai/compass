from __future__ import annotations

from typing import List, Tuple, Union

import jax
import jax.numpy as jnp
from numpy.random import RandomState
from sklearn.cluster import KMeans


def compute_cvt_centroids(
    num_descriptors: int,
    num_init_cvt_samples: int,
    num_centroids: int,
    minval: Union[float, List[float]],
    maxval: Union[float, List[float]],
    random_key: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute centroids for CVT tesselation.
    Args:
        num_descriptors: number od scalar descriptors
        num_init_cvt_samples: number of sampled point to be sued for clustering to
            determine the centroids. The larger the number of centroids and the
            number of descriptors, the higher this value must be (e.g. 100000 for
            1024 centroids and 100 descriptors).
        num_centroids: number of centroids
        minval: minimum descriptors value
        maxval: maximum descriptors value
        random_key: a jax PRNG random key
    Returns:
        the centroids with shape (num_centroids, num_descriptors)
        random_key: an updated jax PRNG random key
    """
    minval = jnp.array(minval)
    maxval = jnp.array(maxval)

    # assume here all values are in [0, 1] and rescale later
    random_key, subkey = jax.random.split(random_key)
    x = jax.random.uniform(key=subkey, shape=(num_init_cvt_samples, num_descriptors))

    # compute k means
    random_key, subkey = jax.random.split(random_key)
    k_means = KMeans(
        init="k-means++",
        n_clusters=num_centroids,
        n_init=1,
        random_state=RandomState(subkey),
    )
    k_means.fit(x)
    centroids = k_means.cluster_centers_
    # rescale now
    return jnp.asarray(centroids) * (maxval - minval) + minval, random_key
