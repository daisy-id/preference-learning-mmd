import numpy as np

def chushizhi(nOfNumericalValue, GammaOfNumericalValue, nOfLinguisticValue, GammaOfLinguisticValue, q, KI, KD, delta_max):
    u_k = np.zeros((nOfNumericalValue, GammaOfNumericalValue), dtype=np.float64)
    u_k_middle = np.zeros((nOfLinguisticValue, max(GammaOfLinguisticValue)), dtype=np.float64)
    u_k_middle[-1, -3:] = np.nan
    u_middle = np.array([0, 0.3, 0.45, 1])

    u_k_MaxValue = [0.0193, 0.0901, 0.1101, 0.2254, 0.1685, 0.0894, 0.1026, 0.1314]
    u_k_middle_MaxValue = [0.0429, 0.0226]

    for k in range(0, KI):
        u_k[k, GammaOfNumericalValue-1] = u_k_MaxValue[k]
        u_k_interval = u_k_MaxValue[k] / (GammaOfNumericalValue - 1)
        for r in range(1, GammaOfNumericalValue-1):
            u_k[k, r] = u_k[k, r-1] + u_k_interval
    for k in range(KI, KD):
        u_k[k, 0] = u_k_MaxValue[k]
        u_k_interval = u_k_MaxValue[k] / (GammaOfNumericalValue - 1)
        for r in range(1, GammaOfNumericalValue - 1):
            u_k[k, r] = u_k[k, r - 1] - u_k_interval
    for k in range(KD, nOfNumericalValue):
        u_k[k, delta_max-1] = u_k_MaxValue[k]
        for r in range(1, delta_max-1):
            u_k_interval = u_k_MaxValue[k] / (delta_max - 1)
            u_k[k, r] = u_k[k, r - 1] + u_k_interval
        for r in range(delta_max, GammaOfNumericalValue-1):
            u_k_interval = u_k_MaxValue[k] / (GammaOfNumericalValue - delta_max)
            u_k[k, r] = u_k[k, r - 1] - u_k_interval

    for k in range(0, nOfLinguisticValue):
        u_k_middle[k, GammaOfLinguisticValue[k] - 1] = u_k_middle_MaxValue[k]
        u_k_middle_interval = u_k_middle_MaxValue[k]/(GammaOfLinguisticValue[k] - 1)
        for r in range(1,  GammaOfLinguisticValue[k] - 1):
            u_k_middle[k, r] = u_k_middle[k, r-1] + u_k_middle_interval
    return u_k, u_k_middle, u_middle


def chushizhi_left_right(u_k_middle, u_middle, nOfLinguisticValue, GammaOfLinguisticValue, q):
    u_k_left = np.zeros((nOfLinguisticValue, max(GammaOfLinguisticValue)), dtype=np.float64)
    u_k_right = np.zeros((nOfLinguisticValue, max(GammaOfLinguisticValue)), dtype=np.float64)
    u_k_left[-1, -3:] = np.nan
    u_k_right[-1, -3:] = np.nan
    u_left = np.zeros(q, dtype=np.float64)
    u_right = np.zeros(q, dtype=np.float64)

    for k in range(nOfLinguisticValue):
        u_k_left[k, 0] = u_k_left[k, 1] = u_k_middle[k, 0]
        for r in range(1, GammaOfLinguisticValue[k]-1):
            u_k_left[k, r+1] = u_k_right[k, r-1] = u_k_middle[k, r]
        u_k_right[k, GammaOfLinguisticValue[k]-2] = u_k_right[k, GammaOfLinguisticValue[k]-1] = u_k_middle[k, GammaOfLinguisticValue[k]-1]

    for g in range(1, q-1):
        u_left[0] = u_left[1] = u_middle[0]
        u_left[g+1] = u_right[g-1] = u_middle[g]
        u_right[q-2] = u_right[q-1] = u_middle[q-1]
    return u_k_left, u_k_right, u_left, u_right


