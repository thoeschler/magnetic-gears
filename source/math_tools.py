import numpy as np

def get_perpendicular(vec, normalized=True):
    # create random unit vector different from vec
    while True:
        rand = np.random.rand(3)
        if not np.allclose(rand, vec):
            break
    # create vector perpendicular to vec
    perp_vec = np.cross(vec, rand)
    if normalized:
        perp_vec /= np.linalg.norm(perp_vec)
    assert np.isclose(np.dot(vec, perp_vec), 0.)
    return perp_vec