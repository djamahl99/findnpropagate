import numpy as np
import random

def ransac(data, model, is_inlier, sample_size, max_iterations, goal_inliers, random_seed=75, debug=False):
    best_ic = 0
    best_model = None
    random.seed(random_seed)

    data = list(data)
    for i in range(max_iterations):
        s = random.sample(data, int(sample_size))
        m = model(s)
        ic = 0
        for j in range(len(data)):
              if is_inlier(m, data[j]):
                      ic += 1

        if debug:
              print('Coeffs:', m, '# inliers:', ic)

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if goal_inliers and ic > goal_inliers:
                  break
    if debug: 
          print('Took iterations:', i+1, 'Best model coeffs:', best_model, 'Inliers covered:', best_ic)

    return best_model, best_ic

def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def model(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

def fit_plane(data: np.ndarray, max_iterations=5, sample_size=10, threshold=0.1, goal_inliers_p=0.5):

    # number of points we want to sample, 3 is minimum, but we can choose more for better fitting.
    # sample_size = 10
    # distance threshold for choosing inliers
    # threshold = 0.1
    # minimum numbers of inliers we need to have, we can ignore this parameter by setting None
    goal_inliers = data.shape[0] * goal_inliers_p

    coeff, _ = ransac(data[:, :3], model, lambda x, y: is_inlier(x, y, threshold), sample_size, max_iterations, goal_inliers)
    proj = data[:,0] * coeff[0] + data[:,1] * coeff[1] + data[:,2] * coeff[2] + coeff[3]

    ground_pts_mask = np.abs(proj) <= threshold
    
    return ground_pts_mask