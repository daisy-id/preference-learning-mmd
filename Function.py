import random
import torch
import numpy as np
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def numerical_value_breakpoint_matrix(nOfNumericalValue, MatrixOfNumericalValue, GammaOfNumericalValue):
    NumericalValueBreakpoint = np.zeros((nOfNumericalValue, GammaOfNumericalValue), dtype=np.float64)

    for col in range(nOfNumericalValue):
        MinOfNumericalValue = np.min(MatrixOfNumericalValue[:, col], axis=0).astype(np.float64)
        MaxOfNumericalValue = np.max(MatrixOfNumericalValue[:, col], axis=0).astype(np.float64)
        NumericalValueBreakpoint[col, :] = MinOfNumericalValue + np.arange(GammaOfNumericalValue).astype(np.float64) * (MaxOfNumericalValue - MinOfNumericalValue) / (GammaOfNumericalValue - 1)
        # print(f"Column {col}: Min = {MinOfNumericalValue}, Max = {MaxOfNumericalValue}, Breakpoints = {NumericalValueBreakpoint[col, :]}")

    return NumericalValueBreakpoint


def linguistic_value_breakpoint_matrix(nOfLinguisticValue, MatrixOfLinguisticValue, GammaOfLinguisticValue):
    max_gamma = max(GammaOfLinguisticValue)
    LinguisticValueBreakpoint = np.zeros((nOfLinguisticValue, max_gamma), dtype=np.float64)

    for col in range(nOfLinguisticValue):
        MinOfLinguisticValue = np.min(MatrixOfLinguisticValue[:, col], axis=0).astype(np.float64)
        MaxOfLinguisticValue = np.max(MatrixOfLinguisticValue[:, col], axis=0).astype(np.float64)
        LinguisticValueBreakpoint[col, :GammaOfLinguisticValue[col]] = MinOfLinguisticValue + np.arange(GammaOfLinguisticValue[col]).astype(np.float64) * (MaxOfLinguisticValue -
                                                                                                                                                          MinOfLinguisticValue) / (GammaOfLinguisticValue[col] - 1)

    return LinguisticValueBreakpoint


def numerical_value_utility_matrix(u_k, m, nOfNumericalValue, MatrixOfNumericalValue, GammaOfNumericalValue, NumericalValueBreakpoint):
    NumericalValueUtilityMatrix = np.zeros((m, nOfNumericalValue), dtype=np.float64)
    NumericalValueBreakpointUtilityMatrix = np.zeros((m, nOfNumericalValue, 2), dtype=np.float64)

    for col in range(nOfNumericalValue):
        for r in range(GammaOfNumericalValue - 1):
            low, upper = NumericalValueBreakpoint[col, r], NumericalValueBreakpoint[col, r + 1]
            mask = (low <= MatrixOfNumericalValue[:, col]) & (MatrixOfNumericalValue[:, col] <= upper)
            values = u_k[col, r] + (MatrixOfNumericalValue[:, col] - low) / (upper - low) * (u_k[col, r + 1] - u_k[col, r])
            NumericalValueUtilityMatrix[:, col] = np.where(mask, values, NumericalValueUtilityMatrix[:, col])
            NumericalValueBreakpointUtilityMatrix[:, col, 0] = np.where(mask, u_k[col, r], NumericalValueBreakpointUtilityMatrix[:, col, 0])
            NumericalValueBreakpointUtilityMatrix[:, col, 1] = np.where(mask, u_k[col, r + 1], NumericalValueBreakpointUtilityMatrix[:, col, 1])
    # print("NumericalValueBreakpointUtilityMatrix[:, :, 0]:\n", NumericalValueBreakpointUtilityMatrix[:, :, 0])

    return NumericalValueUtilityMatrix, NumericalValueBreakpointUtilityMatrix


def linguistic_value_utility_matrix(u_k_left, u_k_middle, u_k_right, m, nOfLinguisticValue, MatrixOfLinguisticValue, GammaOfLinguisticValue, LinguisticValueBreakpoint):
    LinguisticValueUtilityLeftMatrix = np.zeros((m, nOfLinguisticValue), dtype=np.float64)
    LinguisticValueUtilityMiddleMatrix = np.zeros((m, nOfLinguisticValue), dtype=np.float64)
    LinguisticValueUtilityRightMatrix = np.zeros((m, nOfLinguisticValue), dtype=np.float64)

    for k in range(nOfLinguisticValue):
        for r in range(GammaOfLinguisticValue[k]):
            Breakpoint = LinguisticValueBreakpoint[k, r]
            mask = (MatrixOfLinguisticValue[:, k] == Breakpoint)
            LinguisticValueUtilityLeftMatrix[:, k] = np.where(mask, u_k_left[k, r], LinguisticValueUtilityLeftMatrix[:, k])
            LinguisticValueUtilityMiddleMatrix[:, k] = np.where(mask, u_k_middle[k, r], LinguisticValueUtilityMiddleMatrix[:, k])
            LinguisticValueUtilityRightMatrix[:, k] = np.where(mask, u_k_right[k, r], LinguisticValueUtilityRightMatrix[:, k])

    return LinguisticValueUtilityLeftMatrix, LinguisticValueUtilityMiddleMatrix, LinguisticValueUtilityRightMatrix


def linguistic_value_sum_utility_list(LinguisticValueUtilityLeftMatrix, LinguisticValueUtilityMiddleMatrix, LinguisticValueUtilityRightMatrix):
    LinguisticValueSumUtilityLeftList = np.sum(LinguisticValueUtilityLeftMatrix, axis=1, dtype=np.float64)
    LinguisticValueSumUtilityMiddleList = np.sum(LinguisticValueUtilityMiddleMatrix, axis=1, dtype=np.float64)
    LinguisticValueSumUtilityRightList = np.sum(LinguisticValueUtilityRightMatrix, axis=1, dtype=np.float64)


    return LinguisticValueSumUtilityLeftList, LinguisticValueSumUtilityMiddleList, LinguisticValueSumUtilityRightList


def alternative_value_utility_matrix(NumericalValueUtilityMatrix, LinguisticValueSumUtilityLeftList, LinguisticValueSumUtilityMiddleList, LinguisticValueSumUtilityRightList):

    combined_left = np.sum(NumericalValueUtilityMatrix, axis=1, dtype=np.float64) + LinguisticValueSumUtilityLeftList
    combined_middle = np.sum(NumericalValueUtilityMatrix, axis=1, dtype=np.float64) + LinguisticValueSumUtilityMiddleList
    combined_right = np.sum(NumericalValueUtilityMatrix, axis=1, dtype=np.float64) + LinguisticValueSumUtilityRightList

    AlternativeValueUtilityLeftMatrix = combined_left.reshape(-1, 1)
    AlternativeValueUtilityMiddleMatrix = combined_middle.reshape(-1, 1)
    AlternativeValueUtilityRightMatrix = combined_right.reshape(-1, 1)
    return AlternativeValueUtilityLeftMatrix, AlternativeValueUtilityMiddleMatrix, AlternativeValueUtilityRightMatrix

def class_value_utility_matrix(u_left, u_middle, u_right, m, MatrixOfClassValue, q, ClassValueBreakpoint):
    ClassValueUtilityLeftMatrix = np.zeros((m, 1), dtype=np.float64)
    ClassValueUtilityMiddleMatrix = np.zeros((m, 1), dtype=np.float64)
    ClassValueUtilityRightMatrix = np.zeros((m, 1), dtype=np.float64)

    for g in range(q):
        Breakpoint = ClassValueBreakpoint[g]
        mask = (MatrixOfClassValue[:, 0] == Breakpoint)
        ClassValueUtilityLeftMatrix[:, 0] = np.where(mask, u_left[g], ClassValueUtilityLeftMatrix[:, 0])
        ClassValueUtilityMiddleMatrix[:, 0] = np.where(mask, u_middle[g], ClassValueUtilityMiddleMatrix[:, 0])
        ClassValueUtilityRightMatrix[:, 0] = np.where(mask, u_right[g], ClassValueUtilityRightMatrix[:, 0])

    return ClassValueUtilityLeftMatrix, ClassValueUtilityMiddleMatrix, ClassValueUtilityRightMatrix

def membership_function(x, u_left, u_middle, u_right):
    x = np.array(x)

    if u_left == u_middle == u_right:
        return np.where(x == u_middle, 1.0, 0.0)
    elif u_left == u_middle:
        mask = (u_middle <= x) & (x < u_right)
        return np.where(mask, (u_right - x) / (u_right - u_middle), 0.0)
    elif u_middle == u_right:
        mask = (u_left <= x) & (x < u_middle)
        return np.where(mask, (x - u_left) / (u_middle - u_left), 0.0)
    else:
        membership = np.zeros_like(x, dtype=np.float32)
        mask1 = (x >= u_left) & (x < u_middle)
        membership[mask1] = (x[mask1] - u_left) / (u_middle - u_left)
        mask2 = (x > u_middle) & (x < u_right)
        membership[mask2] = (u_right - x[mask2]) / (u_right - u_middle)
        return membership


def alternative_membership(params):
    alternative_left, alternative_middle, alternative_right, class_left, class_middle, class_right = params
    points = sorted([alternative_left, alternative_middle, alternative_right, class_left, class_middle, class_right])
    #print("points :\n", points )
    if alternative_left < class_right and class_left < alternative_right:
        total_integral = 0.0
        for p in range(len(points) - 1):
            x_i, x_next = points[p], points[p + 1]
            x = np.linspace(x_i, x_next, 100)
            y = membership_function(x, alternative_left, alternative_middle, alternative_right) * membership_function(x, class_left, class_middle, class_right)
            integral = np.trapz(y, x)

            total_integral += integral
    else:
        total_integral = 0.0
    #print("total_integral:\n", total_integral)

    return total_integral


def alternative_membership_list(m, AlternativeValueUtilityLeftMatrix, AlternativeValueUtilityMiddleMatrix, AlternativeValueUtilityRightMatrix, ClassValueUtilityLeftMatrix,
                               ClassValueUtilityMiddleMatrix, ClassValueUtilityRightMatrix):

    params_matrix = np.zeros((m, 6), dtype=np.float64)
    params_matrix[:, 0] = AlternativeValueUtilityLeftMatrix[:, 0]
    params_matrix[:, 1] = AlternativeValueUtilityMiddleMatrix[:, 0]
    params_matrix[:, 2] = AlternativeValueUtilityRightMatrix[:, 0]
    params_matrix[:, 3] = ClassValueUtilityLeftMatrix[:, 0]
    params_matrix[:, 4] = ClassValueUtilityMiddleMatrix[:, 0]
    params_matrix[:, 5] = ClassValueUtilityRightMatrix[:, 0]
    #print("params_matrix:\n", params_matrix)

    AlternativeSumMembershipList = np.array([alternative_membership(params) for params in params_matrix])

    AlternativeAvMembershipList= AlternativeSumMembershipList / (params_matrix[:, 2]-params_matrix[:, 0])
    #print("AlternativeAvMembershipList:\n", AlternativeAvMembershipList)

    return params_matrix, AlternativeSumMembershipList, AlternativeAvMembershipList


def loss_function(AlternativeAvMembershipList):
    loss = np.mean(AlternativeAvMembershipList)
    return loss


def correct(u_k, u_k_left, u_k_middle, u_k_right, u_left, u_middle, u_right, m, MatrixOfNumericalValue, MatrixOfLinguisticValue, nOfNumericalValue, GammaOfNumericalValue,
             nOfLinguisticValue, GammaOfLinguisticValue, NumericalValueBreakpoint, LinguisticValueBreakpoint, q, Class4, Class3, Class2, Class1):

    NumericalValueUtilityMatrix, NumericalValueBreakpointUtilityMatrix = numerical_value_utility_matrix(u_k, m, nOfNumericalValue, MatrixOfNumericalValue, GammaOfNumericalValue,
                                                                                                        NumericalValueBreakpoint)
    LinguisticValueUtilityLeftMatrix, LinguisticValueUtilityMiddleMatrix, LinguisticValueUtilityRightMatrix = linguistic_value_utility_matrix(u_k_left, u_k_middle, u_k_right, m,
                                                                                                                                              nOfLinguisticValue, MatrixOfLinguisticValue,
                                                                                                                                              GammaOfLinguisticValue, LinguisticValueBreakpoint)
    LinguisticValueSumUtilityLeftList, LinguisticValueSumUtilityMiddleList, LinguisticValueSumUtilityRightList = linguistic_value_sum_utility_list(LinguisticValueUtilityLeftMatrix,
                                                                                                                                                   LinguisticValueUtilityMiddleMatrix,
                                                                                                                                                   LinguisticValueUtilityRightMatrix)
    AlternativeValueUtilityLeftMatrix, AlternativeValueUtilityMiddleMatrix, AlternativeValueUtilityRightMatrix = alternative_value_utility_matrix(NumericalValueUtilityMatrix,
                                                                                                                                                  LinguisticValueSumUtilityLeftList,
                                                                                                                                                  LinguisticValueSumUtilityMiddleList,
                                                                                                                                                  LinguisticValueSumUtilityRightList)


    params_matrix = np.zeros((m * q, 6))
    params_matrix[:, 0] = np.repeat(AlternativeValueUtilityLeftMatrix[:, 0], q)
    params_matrix[:, 1] = np.repeat(AlternativeValueUtilityMiddleMatrix[:, 0], q)
    params_matrix[:, 2] = np.repeat(AlternativeValueUtilityRightMatrix[:, 0], q)
    params_matrix[:, 3] = np.tile(u_left, m)
    params_matrix[:, 4] = np.tile(u_middle, m)
    params_matrix[:, 5] = np.tile(u_right, m)


    params_matrix_df = pd.DataFrame(params_matrix)
    file_path = 'params_matrix.xlsx'
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        params_matrix_df.to_excel(writer, sheet_name='params_matrix', index=False)

    membership_values = np.apply_along_axis(alternative_membership, 1, params_matrix)
    denominator = params_matrix[:, 2] - params_matrix[:, 0]
    final_membership_values = membership_values / denominator
    MembershipMatrix = final_membership_values.reshape(m, q)

    '''
    MembershipMatrix = np.zeros((m, q))
    params = [0] * 6
    for i in range(m):
        for g in range(q):
            params[0] = AlternativeValueUtilityLeftMatrix[i, 0]
            params[1] = AlternativeValueUtilityMiddleMatrix[i, 0]
            params[2] = AlternativeValueUtilityRightMatrix[i, 0]
            params[3] = u_left[g]
            params[4] = u_middle[g]
            params[5] = u_right[g]
            MembershipMatrix[i, g] = alternative_membership(params)/(params_matrix[:, 2]-params_matrix[:, 0])
    MembershipMatrix_df = pd.DataFrame(MembershipMatrix)
    '''
    #print("MembershipMatrix:\n", MembershipMatrix)


    max_values = np.max(MembershipMatrix, axis=1, keepdims=True)
    ReMembershipMatrix = (MembershipMatrix == max_values).astype(int)
    ReMembershipMatrix[np.all(np.isclose(MembershipMatrix, 0.0), axis=1), :] = 0

    '''
    if np.all(MembershipMatrix == 0):
        ReMembershipMatrix = np.zeros_like(MembershipMatrix)
    '''

    '''
    max_values = np.max(MembershipMatrix, axis=1, keepdims=True)
    ReMembershipMatrix = (MembershipMatrix == max_values).astype(int)
    '''

    ranges = [
        (0, Class4),
        (Class4, Class4 + Class3),
        (Class4 + Class3, Class4 + Class3 + Class2),
        (Class4 + Class3 + Class2, Class4 + Class3 + Class2 + Class1)
    ]

    correct_predictions = [np.sum(ReMembershipMatrix[start:end, i]) for i, (start, end) in enumerate(ranges)]
    sum_correct_predictions = np.sum(correct_predictions)

    return MembershipMatrix, correct_predictions, sum_correct_predictions


def normalized_params(u_k, u_k_middle):

    u_k_diff = np.max(u_k, axis=1) - np.min(u_k, axis=1)
    u_k_middle_diff = np.nanmax(u_k_middle, axis=1) - np.min(u_k_middle, axis=1)

    sum = np.sum(u_k_diff) + np.nansum(u_k_middle_diff)

    u_k_normalized = (u_k - np.min(u_k, axis=1).reshape(-1, 1)) / sum
    u_k_middle_normalized = (u_k_middle - np.nanmin(u_k_middle, axis=1).reshape(-1, 1)) / sum

    return u_k_normalized, u_k_middle_normalized

def normalized_class(u_middle):
    u_middle_diff = np.max(u_middle) - np.min(u_middle)
    u_middle_normalized = (u_middle - np.min(u_middle)) / u_middle_diff
    return u_middle_normalized

def left_right(u_k_middle, u_middle, nOfLinguisticValue, GammaOfLinguisticValue, q):
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

def get_params(KI, KD, KA, delta_max, nOfNumericalValue, GammaOfNumericalValue, u_k):

    u_ki = np.zeros((nOfNumericalValue, GammaOfNumericalValue))
    u_kd = np.zeros((nOfNumericalValue, GammaOfNumericalValue))
    epsilon_k = np.zeros((nOfNumericalValue, 1))

    max = 0.1
    for k in range(0, KI):
        epsilon_k[k, 0] = random.uniform(0.01 * GammaOfNumericalValue, max)
        u_kd[k, 0] = epsilon_k[k, 0]
        u_ki[k, GammaOfNumericalValue-1] = u_k[k, GammaOfNumericalValue-1] + epsilon_k[k, 0] - u_kd[k, GammaOfNumericalValue-1]
        for r in range(1, GammaOfNumericalValue-1):
            u_kd[k, r] = random.uniform(0.01 * (GammaOfNumericalValue-r), u_kd[k, r-1]-0.0001)
            u_ki[k, r] = u_k[k, r] + epsilon_k[k, 0] - u_kd[k, r]
    for k in range(KI, KD):
        epsilon_k[k, 0] = random.uniform(0.01 * GammaOfNumericalValue, max)
        u_ki[k, GammaOfNumericalValue - 1] = epsilon_k[k, 0]
        u_kd[k, 0] = u_k[k, 0] + epsilon_k[k, 0] - u_ki[k, 0]
        for r in range(1, GammaOfNumericalValue-1):
            u_ki[k, GammaOfNumericalValue-1-r] = random.uniform(0.01 * (GammaOfNumericalValue-r), u_ki[k, GammaOfNumericalValue-r]-0.0001)
            u_kd[k, GammaOfNumericalValue-1-r] = u_k[k, GammaOfNumericalValue-1-r] + epsilon_k[k, 0] - u_ki[k, GammaOfNumericalValue-1-r]
    for k in range(KD, KA):
        if u_k[k, 0] == 0 and u_k[k, GammaOfNumericalValue-1] > 0:
            epsilon_k[k, 0] = random.uniform(0.01 * GammaOfNumericalValue, max)
            u_kd[k, 0] = epsilon_k[k, 0]
            u_ki[k, GammaOfNumericalValue - 1] = u_k[k, GammaOfNumericalValue - 1] + epsilon_k[k, 0] - u_kd[k, GammaOfNumericalValue-1]
            for r in range(1, GammaOfNumericalValue - 1):
                u_kd[k, r] = random.uniform(0.01 * (GammaOfNumericalValue - r), u_kd[k, r - 1] - 0.0001)
                u_ki[k, r] = u_k[k, r] + epsilon_k[k, 0] - u_kd[k, r]
        elif u_k[k, 0] > 0 and u_k[k, GammaOfNumericalValue-1] == 0:
            epsilon_k[k, 0] = random.uniform(0.01 * GammaOfNumericalValue, max)
            u_ki[k, GammaOfNumericalValue - 1] = epsilon_k[k, 0]
            u_kd[k, 0] = u_k[k, 0] + epsilon_k[k, 0] - u_ki[k, 0]
            for r in range(1, GammaOfNumericalValue - 1):
                u_ki[k, GammaOfNumericalValue - 1 - r] = random.uniform(0.01 * (GammaOfNumericalValue - r), u_ki[k, GammaOfNumericalValue - r] - 0.0001)
                u_kd[k, GammaOfNumericalValue - 1 - r] = u_k[k, GammaOfNumericalValue - 1 - r] + epsilon_k[k, 0] - u_ki[k, GammaOfNumericalValue - 1 - r]
        elif u_k[k, 0] == 0 and u_k[k, GammaOfNumericalValue-1] == 0:
            epsilon_k[k, 0] = random.uniform(u_k[k, delta_max-1] + 0.01 * delta_max, 0.2)
            u_kd[k, 0] = epsilon_k[k, 0]
            u_ki[k, GammaOfNumericalValue - 1] = epsilon_k[k, 0]
            for r in range(1, delta_max):
                u_kd[k, r] = random.uniform(u_k[k, delta_max-1] + 0.01 * (delta_max-r), u_kd[k, r-1]-0.0001)
                u_ki[k, r] = u_k[k, r] + epsilon_k[k, 0] - u_kd[k, r]
            for r in range(delta_max, GammaOfNumericalValue-1):
                u_ki[k, r] = random.uniform(u_ki[k, r-1] + 0.0001, u_ki[k, GammaOfNumericalValue - 1]-0.0002*(GammaOfNumericalValue-r))
                u_kd[k, r] = u_k[k, r] + epsilon_k[k, 0] - u_ki[k, r]

    return u_ki, u_kd, epsilon_k





