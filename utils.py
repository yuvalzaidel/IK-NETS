import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import mean_squared_error

#import nengo_loihi
import nengo
from nengo.processes import WhiteSignal
from nengo.dists import Distribution, UniformHypersphere
from scipy.special import beta, betainc, betaincinv
from scipy.linalg import svd
import scipy.special

def generate_scaling_functions(means, scales):
    
    def scale_down(t, x):
        index = np.where(scales != 0)
        x[index] = (x[index] - means[index]) / scales[index]
        index = np.where(scales == 0)
        x[index] = x[index] - means[index]
        return x
        
    scale_up = lambda x: x * scales + means

    return scale_down, scale_up
    
    
class Rd(Distribution):

    def __repr__(self):
        return "%s()" % (type(self).__name__)

    def sample(self, n, d=1, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        if d == 1:
            # Tile the points optimally. 
            return np.linspace(1.0 / n, 1, n)[:, None]
        if d is None or not isinstance(d, (int, np.integer)) or d < 1:
            raise ValueError("d (%d) must be positive integer" % d)
        return _rd_generate(n, d)

class ScatteredHypersphere(UniformHypersphere):

    def __init__(self, surface, base=Rd()):
        super(ScatteredHypersphere, self).__init__(surface)
        self.base = base

    def __repr__(self):
        return "%s(surface=%r, base=%r)" % (
            type(self).__name__,
            self.surface,
            self.base,
        )

    def sample(self, n, d=1, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        if d == 1:
            return super(ScatteredHypersphere, self).sample(n, d, rng)

        if self.surface:
            samples = self.base.sample(n, d - 1, rng)
            radius = 1.0
        else:
            samples = self.base.sample(n, d, rng)
            samples, radius = samples[:, :-1], samples[:, -1:] ** (1.0 / d)

        mapped = spherical_transform(samples)

        # radius adjustment for ball versus sphere, and a random rotation
        rotation = random_orthogonal(d, rng=rng)
        return np.dot(mapped * radius, rotation)


class SphericalCoords(Distribution):

    def __init__(self, m):
        super(SphericalCoords, self).__init__()
        self.m = m

    def __repr__(self):
        return "%s(%r)" % (type(self).__name__, self.m)

    def sample(self, n, d=None, rng=np.random):
        """Samples ``n`` points in ``d`` dimensions."""
        shape = self._sample_shape(n, d)
        y = rng.uniform(size=shape)
        return self.ppf(y)

    def pdf(self, x):
        """Evaluates the PDF along the values ``x``."""
        return np.pi * np.sin(np.pi * x) ** (self.m - 1) / beta(self.m / 2.0, 0.5)

    def cdf(self, x):
        """Evaluates the CDF along the values ``x``."""
        y = 0.5 * betainc(self.m / 2.0, 0.5, np.sin(np.pi * x) ** 2)
        return np.where(x < 0.5, y, 1 - y)

    def ppf(self, y):
        """Evaluates the inverse CDF along the values ``x``."""
        y_reflect = np.where(y < 0.5, y, 1 - y)
        z_sq = betaincinv(self.m / 2.0, 0.5, 2 * y_reflect)
        x = np.arcsin(np.sqrt(z_sq)) / np.pi
        return np.where(y < 0.5, x, 1 - x)


def random_orthogonal(d, rng=None):

    rng = np.random if rng is None else rng
    m = UniformHypersphere(surface=True).sample(d, d, rng=rng)
    u, s, v = svd(m)
    return np.dot(u, v)

def _rd_generate(n, d, seed=0.5):

    def gamma(d, n_iter=20):
        """Newton-Raphson-Method to calculate g = phi_d."""
        x = 1.0
        for _ in range(n_iter):
            x -= (x ** (d + 1) - x - 1) / ((d + 1) * x ** d - 1)
        return x

    g = gamma(d)
    alpha = np.zeros(d)
    for j in range(d):
        alpha[j] = (1 / g) ** (j + 1) % 1

    z = np.zeros((n, d))
    z[0] = (seed + alpha) % 1
    for i in range(1, n):
        z[i] = (z[i - 1] + alpha) % 1

    return z

def spherical_transform(samples):

    samples = np.asarray(samples)
    samples = samples[:, None] if samples.ndim == 1 else samples
    coords = np.empty_like(samples)
    n, d = coords.shape

    # inverse transform method (section 1.5.2)
    for j in range(d):
        coords[:, j] = SphericalCoords(d - j).ppf(samples[:, j])

    # spherical coordinate transform
    mapped = np.ones((n, d + 1))
    i = np.ones(d)
    i[-1] = 2.0
    s = np.sin(i[None, :] * np.pi * coords)
    c = np.cos(i[None, :] * np.pi * coords)
    mapped[:, 1:] = np.cumprod(s, axis=1)
    mapped[:, :-1] *= c
    return mapped


def get_encoders (n_neurons, dimensions):
    encoders_dist = ScatteredHypersphere(surface=True)
    
encoders_dist = ScatteredHypersphere(surface=True)



def get_intercepts(n_neurons, dimensions):

    triangular = np.random.triangular(left=0.35, 
                                      mode=0.45, 
                                      right=0.55, 
                                      size=n_neurons)
                                      
    intercepts = nengo.dists.CosineSimilarity(dimensions + 2).ppf(1 - triangular)
    return intercepts



def calc_J(q):

    c0 = np.cos(q[0])
    c1 = np.cos(q[1])
    c2 = np.cos(q[2])
    c3 = np.cos(q[3])
    c4 = np.cos(q[4])
    
    s0 = np.sin(q[0])
    s1 = np.sin(q[1])
    s2 = np.sin(q[2])
    s3 = np.sin(q[3])
    s4 = np.sin(q[4])

    s12  = np.sin(q[1] + q[2])
    c12  = np.cos(q[1] + q[2])
    
    J = np.zeros((3,5))
    J[0,0]= 0.208*((s0*s1*c2 + s0*s2*c1)*c3 + s3*c0)*s4 + 0.208*(s0*s1*s2 - s0*c1*c2)*c4 + 0.299*s0*s1*s2 + 0.3*s0*s1 - 0.299*s0*c1*c2 - 0.06*s0*c1
    J[0,1]= 0.208*(s1*s2*c0 - c0*c1*c2)*s4*c3 + 0.208*(-s1*c0*c2 - s2*c0*c1)*c4 - 0.299*s1*c0*c2 - 0.06*s1*c0 - 0.299*s2*c0*c1 - 0.3*c0*c1
    J[0,2]= 0.208*(s1*s2*c0 - c0*c1*c2)*s4*c3 + 0.208*(-s1*c0*c2 - s2*c0*c1)*c4 - 0.299*s1*c0*c2 - 0.299*s2*c0*c1
    J[0,3]= 0.208*(-(-s1*c0*c2 - s2*c0*c1)*s3 + s0*c3)*s4
    J[0,4]= 0.208*((-s1*c0*c2 - s2*c0*c1)*c3 + s0*s3)*c4 - 0.208*(-s1*s2*c0 + c0*c1*c2)*s4
    J[1,0]= 0
    J[1,1]= 0.208*(-s1*s2 + c1*c2)*c4 + 0.208*(-s1*c2 - s2*c1)*s4*c3 - 0.299*s1*s2 - 0.3*s1 + 0.299*c1*c2 + 0.06*c1
    J[1,2]= 0.208*(-s1*s2 + c1*c2)*c4 + 0.208*(-s1*c2 - s2*c1)*s4*c3 - 0.299*s1*s2 + 0.299*c1*c2
    J[1,3]= -0.208*(-s1*s2 + c1*c2)*s3*s4
    J[1,4]= 0.208*(-s1*s2 + c1*c2)*c3*c4 - 0.208*(s1*c2 + s2*c1)*s4
    J[2,0]= 0.208*((s1*c0*c2 + s2*c0*c1)*c3 - s0*s3)*s4 + 0.208*(s1*s2*c0 - c0*c1*c2)*c4 + 0.299*s1*s2*c0 + 0.3*s1*c0 - 0.299*c0*c1*c2 - 0.06*c0*c1
    J[2,1]= 0.208*(-s0*s1*s2 + s0*c1*c2)*s4*c3 + 0.208*(s0*s1*c2 + s0*s2*c1)*c4 + 0.299*s0*s1*c2 + 0.06*s0*s1 + 0.299*s0*s2*c1 + 0.3*s0*c1
    J[2,2]= 0.208*(-s0*s1*s2 + s0*c1*c2)*s4*c3 + 0.208*(s0*s1*c2 + s0*s2*c1)*c4 + 0.299*s0*s1*c2 + 0.299*s0*s2*c1
    J[2,3]= 0.208*(-(s0*s1*c2 + s0*s2*c1)*s3 + c0*c3)*s4
    J[2,4]= 0.208*((s0*s1*c2 + s0*s2*c1)*c3 + s3*c0)*c4 - 0.208*(s0*s1*s2 - s0*c1*c2)*s4
    
    return J