from libs.clustering import VectorWrapper, MedianCut
import numpy as np
import matplotlib.pyplot as plt
import time


def main():
    start = time.time()
    max_n_elem = 100
    med_cut = MedianCut()
    end = time.time()
    print("create MedianCut object: " + str((end - start) * 1000) + " msec")

    start = time.time()
    mu = [0, 0]
    sigma = [[30, 20], [20, 50]]
    data = np.random.multivariate_normal(mu, sigma, 10000)
    end = time.time()
    print("data generation: " + str((end - start) * 1000) + " msec")

    start = time.time()
    vectors = [VectorWrapper(d) for d in data]
    clusters = med_cut.divide(vectors, max_n_elem)
    end = time.time()
    print("clustering : " + str((end - start) * 1000) + " msec")

    start = time.time()
    for cluster in clusters:
        vecs = np.array([vec.vector for vec in cluster])
        plt.scatter(vecs.T[0], vecs.T[1])
    end = time.time()
    print("plotting  : " + str((end - start) * 1000) + " msec")
    plt.show()


if __name__ == '__main__':
    main()
