from libs.clustering import DataOptimizer, VectorWrapper, RandomSelector, KMeansSelector
import numpy as np
import matplotlib.pyplot as plt
import time


def main():
    start = time.time()
    max_data_size = 100
    # selector = DataOptimizer(max_data_size=max_data_size, use_dwave=False, use_amplify=False)
    # selector = RandomSelector(500)
    selector = KMeansSelector(500)
    end = time.time()
    print("create MedianCut object: " + str((end - start) * 1000) + " msec")

    start = time.time()
    mu = [0, 0]
    sigma = [[30, 20], [20, 50]]
    n_data = 60000
    # data = np.random.multivariate_normal(mu, sigma, n_data)
    data = np.random.rand(n_data, 64*32*3*3)
    end = time.time()
    print("data generation: " + str((end - start) * 1000) + " msec")

    start = time.time()
    vectors = [VectorWrapper(d, i) for i, d in enumerate(data)]
    minimum_limit = 1000
    leaved_data, dropped_data = selector.select(vectors, minimum_limit=minimum_limit)
    end = time.time()
    print("clustering : " + str((end - start) * 1000) + " msec")

    # 残ったデータのweightの合計を確認
    w = 0
    for vector in leaved_data:
        w += vector.weight
    print("leaved weight sum:", w)

    start = time.time()
    VectorWrapper.plot_vectors(
        dropped_data, color_as_parent=True, label='dropped')
    VectorWrapper.plot_vectors(leaved_data, label='leaved')
    print("leaved/dropped", len(leaved_data), "/", len(dropped_data))
    plt.legend()
    end = time.time()
    print("plotting  : " + str((end - start) * 1000) + " msec")
    plt.show()


if __name__ == '__main__':
    main()
