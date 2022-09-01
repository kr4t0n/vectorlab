import numpy as np
import vectorlab as vl

n_samples = 10 ** 6
x = np.random.rand(n_samples)
t_digest = vl.stats.TDigest(buffer_size=100)

vl.utils.loading()(t_digest.fit)(x)

quantiles = np.array([.1, .5, .9])
resutls = t_digest.predict(quantiles)
actual_results = np.sort(x)[(quantiles * n_samples).astype(np.int_)]

print(resutls)
print(actual_results)
