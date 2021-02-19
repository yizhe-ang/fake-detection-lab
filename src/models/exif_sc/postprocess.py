import numpy as np
import scipy
import sklearn.cluster


def mean_shift(points_, heat_map, iters=5):
    points = np.copy(points_)
    kdt = scipy.spatial.cKDTree(points)
    eps_5 = np.percentile(
        scipy.spatial.distance.cdist(points, points, metric="euclidean"), 10
    )

    for epis in range(iters):
        for point_ind in range(points.shape[0]):
            point = points[point_ind]
            nearest_inds = kdt.query_ball_point(point, r=eps_5)
            points[point_ind] = np.mean(points[nearest_inds], axis=0)
    val = []
    for i in range(points.shape[0]):
        val.append(
            kdt.count_neighbors(scipy.spatial.cKDTree(np.array([points[i]])), r=eps_5)
        )
    mode_ind = np.argmax(val)
    ind = np.nonzero(val == np.max(val))
    return np.mean(points[ind[0]], axis=0).reshape(heat_map.shape[0], heat_map.shape[1])


def normalized_cut(res):
    sc = sklearn.cluster.SpectralClustering(
        n_clusters=2, n_jobs=-1, affinity="precomputed"
    )
    out = sc.fit_predict(res.reshape((res.shape[0] * res.shape[1], -1)))
    vis = out.reshape((res.shape[0], res.shape[1]))
    return vis
