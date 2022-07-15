from libs.clustering import VectorWrapper, AnnealingSampler
import numpy as np
import matplotlib.pyplot as plt
import time


def main():
    start = time.time()
    n_sample = 100
    limit_coeff = 0.5
    sampler = AnnealingSampler(
        n_sample=n_sample, limit_coeff=limit_coeff, use_dwave=False, use_remote_amplify=False)
    end = time.time()
    print("create MedianCut object: " + str((end - start) * 1000) + " msec")

    start = time.time()
    mu = [0, 0]
    sigma = [[30, 20], [20, 50]]
    n_data = 1000
    data = np.random.multivariate_normal(mu, sigma, n_data)
    end = time.time()
    print("data generation: " + str((end - start) * 1000) + " msec")

    start = time.time()
    near_limit = 7
    vectors = [VectorWrapper(d, i) for i, d in enumerate(data)]
    leaved_data, dropped_data, _, _ = sampler.select(vectors, near_limit)
    end = time.time()
    print("clustering : " + str((end - start) * 1000) + " msec")

    start = time.time()
    VectorWrapper.plot_vectors(
        dropped_data, color_as_parent=True, label='dropped')
    VectorWrapper.plot_vectors(leaved_data, label='leaved')
    print([vec.weight for vec in leaved_data])
    plt.legend()
    end = time.time()
    print("plotting  : " + str((end - start) * 1000) + " msec")
    plt.show()


if __name__ == '__main__':
    main()
