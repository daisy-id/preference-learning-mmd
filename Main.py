import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import os
import time
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from IPython.display import clear_output
from joblib import Parallel, delayed
from Function import numerical_value_breakpoint_matrix, linguistic_value_breakpoint_matrix, \
    numerical_value_utility_matrix, linguistic_value_utility_matrix, linguistic_value_sum_utility_list, alternative_value_utility_matrix, class_value_utility_matrix, \
    alternative_membership_list, loss_function, correct, normalized_params, normalized_class, get_params
from PartialDerivative import alternative_loss_u_k, alternative_loss_u_k_middle, alternative_loss_u_middle
from InitialValue import chushizhi, chushizhi_left_right


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Set parameters
num_cores = -1
iterations = 70
tolerance = 0.0001
GammaOfNumericalValue = 4
delta_max = 2
samples_per_class = 7
np.random.seed(42)

KI = 4
KD = KI+3
KA = KD+1
GammaOfLinguisticValue = [6, 3]
q = 4
ClassValueBreakpoint = [4, 3, 2, 1]

# 读取Excel文件，并转为数组
base_path = './Data'

TrainSet_filename = 'TrainSet_normalized.xlsx'
TestSet_filename = 'TestSet_normalized.xlsx'

MatrixOfTrainSet_template = os.path.join(base_path, TrainSet_filename)
MatrixOfTestSet_template = os.path.join(base_path, TestSet_filename)

MatrixOfTrainSet = pd.read_excel(MatrixOfTrainSet_template)
NumericalValueOfTrainSet = MatrixOfTrainSet.iloc[:, :8].values
LinguisticValueOfTrainSet = MatrixOfTrainSet.iloc[:, 8:10].values
ClassValueOfTrainSet = MatrixOfTrainSet.iloc[:, [10]].values

MatrixOfTestSet = pd.read_excel(MatrixOfTestSet_template)
NumericalValueOfTestSet = MatrixOfTestSet.iloc[:, :8].values
LinguisticValueOfTestSet = MatrixOfTestSet.iloc[:, 8:10].values
ClassValueOfTestSet = MatrixOfTestSet.iloc[:, [10]].values

nOfNumericalValue = NumericalValueOfTrainSet.shape[1]
nOfLinguisticValue = LinguisticValueOfTrainSet.shape[1]

mOfTrainSet = NumericalValueOfTrainSet.shape[0]
mOfTestSet = NumericalValueOfTestSet.shape[0]

#
uniqueOfTrainSet, countsOfTrainSet = np.unique(ClassValueOfTrainSet, return_counts=True)
value_countsOfTrainSet = dict(zip(uniqueOfTrainSet.flatten(), countsOfTrainSet))
uniqueOfTestSet, countsOfTestSet = np.unique(ClassValueOfTestSet, return_counts=True)
value_countsOfTestSet = dict(zip(uniqueOfTestSet.flatten(), countsOfTestSet))

Class4OfTrainSet = value_countsOfTrainSet.get(4, 0)
Class3OfTrainSet = value_countsOfTrainSet.get(3, 0)
Class2OfTrainSet = value_countsOfTrainSet.get(2, 0)
Class1OfTrainSet = value_countsOfTrainSet.get(1, 0)

Class4OfTestSet = value_countsOfTestSet.get(4, 0)
Class3OfTestSet = value_countsOfTestSet.get(3, 0)
Class2OfTestSet = value_countsOfTestSet.get(2, 0)
Class1OfTestSet = value_countsOfTestSet.get(1, 0)


# chushizhi
u_k, u_k_middle, u_middle = chushizhi(nOfNumericalValue, GammaOfNumericalValue, nOfLinguisticValue, GammaOfLinguisticValue, q, KI, KD, delta_max)
u_k_left, u_k_right, u_left, u_right = chushizhi_left_right(u_k_middle, u_middle, nOfLinguisticValue, GammaOfLinguisticValue, q)



NumericalValueBreakpoint = numerical_value_breakpoint_matrix(nOfNumericalValue, NumericalValueOfTrainSet, GammaOfNumericalValue)
LinguisticValueBreakpoint = linguistic_value_breakpoint_matrix(nOfLinguisticValue, LinguisticValueOfTrainSet, GammaOfLinguisticValue)


def compute_grad_u_k(n, gamma, params_matrix, u_k, MatrixOfNumericalValue, NumericalValueBreakpointUtilityMatrix, LinguisticValueSumUtilityLeftList, LinguisticValueSumUtilityMiddleList, LinguisticValueSumUtilityRightList):
    grad_u_k = 0
    for m, params in enumerate(params_matrix):
        U_k = u_k[n, gamma]
        NumericalValue = MatrixOfNumericalValue[m, n]
        low, upper = NumericalValueBreakpointUtilityMatrix[m, n, 0], NumericalValueBreakpointUtilityMatrix[m, n, 1]
        LinguisticValueSumUtilityLeft, LinguisticValueSumUtilityMiddle, LinguisticValueSumUtilityRight = LinguisticValueSumUtilityLeftList[m], LinguisticValueSumUtilityMiddleList[m], LinguisticValueSumUtilityRightList[m]
        grad_u_k += -alternative_loss_u_k(U_k, NumericalValue, low, upper, params, LinguisticValueSumUtilityLeft, LinguisticValueSumUtilityMiddle, LinguisticValueSumUtilityRight)
    grad_u_k = (1/mOfTrainSet) * grad_u_k
    return grad_u_k


def compute_grad_u_k_middle(n, gamma, params_matrix, u_k_middle, LinguisticValueUtilityMiddleMatrix, LinguisticValueSumUtilityLeftList, LinguisticValueSumUtilityMiddleList, LinguisticValueSumUtilityRightList):
    grad_u_k_middle = 0
    for m, params in enumerate(params_matrix):
        U_k_middle = u_k_middle[n, gamma]
        LinguisticValueUtilityMiddle = LinguisticValueUtilityMiddleMatrix[m, n]
        LinguisticValueSumUtilityLeft, LinguisticValueSumUtilityMiddle, LinguisticValueSumUtilityRight = LinguisticValueSumUtilityLeftList[m], LinguisticValueSumUtilityMiddleList[m], LinguisticValueSumUtilityRightList[m]
        grad_u_k_middle += -alternative_loss_u_k_middle(U_k_middle, LinguisticValueUtilityMiddle, params, LinguisticValueSumUtilityLeft, LinguisticValueSumUtilityMiddle, LinguisticValueSumUtilityRight)
    grad_u_k_middle = (1/mOfTrainSet) * grad_u_k_middle
    return grad_u_k_middle


def compute_grad_u_middle(g, params_matrix, u_middle, ClassValueUtilityMiddleMatrix, LinguisticValueSumUtilityLeftList, LinguisticValueSumUtilityMiddleList, LinguisticValueSumUtilityRightList):
    grad_u_middle = 0
    for m, params in enumerate(params_matrix):
        U_middle = u_middle[g]
        ClassValueUtilityMiddle = ClassValueUtilityMiddleMatrix[m, 0]
        LinguisticValueSumUtilityLeft, LinguisticValueSumUtilityMiddle, LinguisticValueSumUtilityRight = LinguisticValueSumUtilityLeftList[m], LinguisticValueSumUtilityMiddleList[m], \
                                                                                                         LinguisticValueSumUtilityRightList[m]
        grad_u_middle += -alternative_loss_u_middle(U_middle, ClassValueUtilityMiddle, params, LinguisticValueSumUtilityLeft, LinguisticValueSumUtilityRight)
    grad_u_middle = (1/mOfTrainSet) * grad_u_middle
    return grad_u_middle


def scale_gradients(GradMatrix, paramter, i, loss_history):
    alpha_b = 0.006
    alpha = alpha_b
    '''
    # 
    alpha_b = 0.006
    alpha = alpha_b  
    # 
    gamma = 1 / 10000
    # 
    N2 = 1

    # 
    count = 0
    N1 = -1
    for i, loss in loss_history:
        if loss > 0.3:
            count += 1
            if count == 1:
                alpha = 0.001
                N1 = i
        if N1 != -1 and i > N1:
            alpha = 0.0001 * (gamma ** (math.floor((i-N1)/N2)*N2))
    '''

    max_value = np.max(np.abs(GradMatrix))
    max_values = np.array([max_value] * GradMatrix.shape[0])

    scaling_factors = []
    for max_val in max_values:
        if max_val == 0:
            digits = 0
        else:
            digits = int(np.log10(max_val)) + 1
        scaling_factor = 1 / (10 ** (digits + 0))
        scaling_factors.append(scaling_factor)
    scaling_factors = np.array(scaling_factors)

    scaled_GradMatrix = GradMatrix * scaling_factors[:, np.newaxis] * alpha

    return scaled_GradMatrix


def project(u_k, u_k_middle, u_middle):
    # u_k
    for k in range(0, KI):
        for r in range(0, GammaOfNumericalValue-1):
            if u_k[k, GammaOfNumericalValue-1-r] <= u_k[k, GammaOfNumericalValue-2-r]:
                u_k[k, GammaOfNumericalValue - 2 - r] = u_k[k, GammaOfNumericalValue-1-r] - 0.0001
    for k in range(KI, KD):
        for r in range(0, GammaOfNumericalValue - 1):
            if u_k[k, r] <= u_k[k, r + 1]:
                u_k[k, r + 1] = u_k[k, r] - 0.0001
    for k in range(KD, KA):
        for r in range(0, delta_max-1):
            if u_k[k, delta_max-1-r] <= u_k[k, delta_max-2-r]:
                u_k[k, delta_max - 2 - r] = u_k[k, delta_max-1-r] - 0.0001

        for r in range(delta_max-1, GammaOfNumericalValue-1):
            if u_k[k, r] <= u_k[k, r + 1]:
                u_k[k, r + 1] = u_k[k, r] - 0.0001

    # u_k_middle
    for k in range(0, nOfLinguisticValue):
        for r in range(0,  GammaOfLinguisticValue[k] - 1):
            if u_k_middle[k, GammaOfLinguisticValue[k]- 1-r] <= u_k_middle[k, GammaOfLinguisticValue[k] - 2-r]:
                u_k_middle[k, GammaOfLinguisticValue[k] - 2 - r] = u_k_middle[k, GammaOfLinguisticValue[k] - 1-r] - 0.01

    # u_middle
    for g in range(0, q-1):
        if u_middle[q-1-g] <= u_middle[q-2-g]:
            u_middle[q-2-g] = u_middle[q-1-g] - 0.01
    return u_k, u_k_middle, u_middle


def left_right(u_k_middle, u_middle):

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


def undersampling_minibatch_generator(data, samples_per_class):  # samples_per_class：每个类别的样本数
    while True:
        undersampled_data = data.groupby('YJDJ_reclass').apply(lambda x: x.sample(samples_per_class)).reset_index(drop=True)
        undersampled_data = undersampled_data.sort_values(by='YJDJ_reclass', ascending=False).reset_index(drop=True)
        NumericalBatch = undersampled_data.iloc[:, :8].values
        LinguisticBatch = undersampled_data.iloc[:, 8:10].values
        ClassBatch = undersampled_data.iloc[:, [10]].values
        mOfBatch = NumericalBatch.shape[0]

        yield mOfBatch, NumericalBatch, LinguisticBatch, ClassBatch


def compute_loss_and_gradient(u_k, u_k_left, u_k_middle, u_k_right, u_left, u_middle, u_right, mOfTrainSet, NumericalValueOfTrainSet, LinguisticValueOfTrainSet, ClassValueOfTrainSet):



    NumericalValueUtilityMatrix, NumericalValueBreakpointUtilityMatrix = numerical_value_utility_matrix(u_k, mOfTrainSet, nOfNumericalValue, NumericalValueOfTrainSet, GammaOfNumericalValue, NumericalValueBreakpoint)
    LinguisticValueUtilityLeftMatrix, LinguisticValueUtilityMiddleMatrix, LinguisticValueUtilityRightMatrix = linguistic_value_utility_matrix(u_k_left, u_k_middle, u_k_right, mOfTrainSet, nOfLinguisticValue, LinguisticValueOfTrainSet, GammaOfLinguisticValue, LinguisticValueBreakpoint)
    LinguisticValueSumUtilityLeftList, LinguisticValueSumUtilityMiddleList, LinguisticValueSumUtilityRightList = linguistic_value_sum_utility_list(LinguisticValueUtilityLeftMatrix, LinguisticValueUtilityMiddleMatrix, LinguisticValueUtilityRightMatrix)
    AlternativeValueUtilityLeftMatrix, AlternativeValueUtilityMiddleMatrix, AlternativeValueUtilityRightMatrix = alternative_value_utility_matrix(NumericalValueUtilityMatrix, LinguisticValueSumUtilityLeftList, LinguisticValueSumUtilityMiddleList, LinguisticValueSumUtilityRightList)



    ClassValueUtilityLeftMatrix, ClassValueUtilityMiddleMatrix, ClassValueUtilityRightMatrix = class_value_utility_matrix(u_left, u_middle, u_right, mOfTrainSet, ClassValueOfTrainSet, q, ClassValueBreakpoint)
    params_matrix, AlternativeSumMembershipList, AlternativeAvMembershipList = alternative_membership_list(mOfTrainSet, AlternativeValueUtilityLeftMatrix, AlternativeValueUtilityMiddleMatrix,
                                                                                                AlternativeValueUtilityRightMatrix, ClassValueUtilityLeftMatrix, ClassValueUtilityMiddleMatrix,
                                                                                                ClassValueUtilityRightMatrix)
    loss = loss_function(AlternativeAvMembershipList)


    GradU_kMatrix = np.zeros((nOfNumericalValue, GammaOfNumericalValue), dtype=np.float32)
    results = Parallel(n_jobs=num_cores)(delayed(compute_grad_u_k)(n, gamma, params_matrix, u_k, NumericalValueOfTrainSet, NumericalValueBreakpointUtilityMatrix, LinguisticValueSumUtilityLeftList, LinguisticValueSumUtilityMiddleList, LinguisticValueSumUtilityRightList) for n in range(nOfNumericalValue) for gamma in range(GammaOfNumericalValue))
    for (n, gamma), result in zip([(n, gamma) for n in range(nOfNumericalValue) for gamma in range(GammaOfNumericalValue)], results):
        GradU_kMatrix[n, gamma] = result

    GradU_k_middleMatrix = np.zeros((nOfLinguisticValue, max(GammaOfLinguisticValue)), dtype=np.float32)
    results = Parallel(n_jobs=num_cores)(delayed(compute_grad_u_k_middle)(n, gamma, params_matrix, u_k_middle, LinguisticValueUtilityMiddleMatrix, LinguisticValueSumUtilityLeftList, LinguisticValueSumUtilityMiddleList, LinguisticValueSumUtilityRightList) for n in range(nOfLinguisticValue) for gamma in range(GammaOfLinguisticValue[n]))
    for (n, gamma), result in zip([(n, gamma) for n in range(nOfLinguisticValue) for gamma in range(GammaOfLinguisticValue[n])], results):
        GradU_k_middleMatrix[n, gamma] = result

    GradU_middleMatrix = np.zeros(q, dtype=np.float32)
    results = Parallel(n_jobs=num_cores)(delayed(compute_grad_u_middle)(g, params_matrix, u_middle, ClassValueUtilityMiddleMatrix, LinguisticValueSumUtilityLeftList, LinguisticValueSumUtilityMiddleList, LinguisticValueSumUtilityRightList) for g in range(q))
    for g, result in enumerate(results):
        GradU_middleMatrix[g] = result


    scaled_GradU_kMatrix = scale_gradients(GradU_kMatrix, u_k, i, loss_history)
    scaled_GradU_k_middleMatrix = scale_gradients(GradU_k_middleMatrix, u_k_middle, i, loss_history)
    scaled_GradU_middleMatrix = scale_gradients(GradU_middleMatrix.reshape(1, -1), u_middle.reshape(1, -1), i, loss_history).flatten()

    u_k += -scaled_GradU_kMatrix
    u_k_middle += -scaled_GradU_k_middleMatrix
    u_middle += -scaled_GradU_middleMatrix

    u_k, u_k_middle, u_middle = project(u_k, u_k_middle, u_middle)


    u_middle[0] = np.min(AlternativeValueUtilityLeftMatrix)
    u_middle[3] = np.max(AlternativeValueUtilityRightMatrix)

    u_k_left, u_k_right, u_left, u_right = left_right(u_k_middle, u_middle)

    u_k1, u_k_middle1 = normalized_params(u_k, u_k_middle)
    u_middle1 = normalized_class(u_middle)
    u_k_left1, u_k_right1, u_left1, u_right1 = left_right(u_k_middle, u_middle)
    u_ki1, u_kd1, epsilon_k1 = get_params(KI, KD, KA, delta_max, nOfNumericalValue, GammaOfNumericalValue, u_k1)
    u_values = {
        "u_ki1": u_ki1,
        "u_kd1": u_kd1,
        "epsilon_k1": epsilon_k1,
        "u_k1": u_k1,
        "u_k_left1": u_k_left1,
        "u_k_middle1": u_k_middle1,
        "u_k_right1": u_k_right1,
        "u_left1": u_left1,
        "u_middle1": u_middle1,
        "u_right1": u_right1
    }

    return loss, u_k, u_k_left, u_k_middle, u_k_right, u_left, u_middle, u_right, u_values


# run

loss_history = []
parameter_history = []

MembershipMatrixOfTrainSet_history = []
CorrectOfTrainSet_history = []
SumCorrectOfTrainSet_history = []

MembershipMatrixOfTestSet_history = []
CorrectOfTestSet_history = []
SumCorrectOfTestSet_history = []

StopCondition_history = []

for i in range(iterations+1):

    MembershipMatrixOfTestSet, CorrectOfTestSet, SumCorrectOfTestSet = correct(u_k, u_k_left, u_k_middle, u_k_right, u_left, u_middle, u_right, mOfTestSet, NumericalValueOfTestSet,
                                                                                LinguisticValueOfTestSet,
                                                                      nOfNumericalValue, GammaOfNumericalValue, nOfLinguisticValue, GammaOfLinguisticValue, NumericalValueBreakpoint,
                                                                       LinguisticValueBreakpoint, q, Class4OfTestSet, Class3OfTestSet, Class2OfTestSet, Class1OfTestSet)
    MembershipMatrixOfTrainSet, CorrectOfTrainSet, SumCorrectOfTrainSet = correct(u_k, u_k_left, u_k_middle, u_k_right, u_left, u_middle, u_right, mOfTrainSet, NumericalValueOfTrainSet,
                                                                                   LinguisticValueOfTrainSet, nOfNumericalValue, GammaOfNumericalValue, nOfLinguisticValue, GammaOfLinguisticValue,
                                                                                   NumericalValueBreakpoint, LinguisticValueBreakpoint, q, Class4OfTrainSet, Class3OfTrainSet, Class2OfTrainSet,
                                                                                   Class1OfTrainSet)

    minibatch_gen = undersampling_minibatch_generator(MatrixOfTrainSet, samples_per_class)
    mOfBatch, NumericalBatch, LinguisticBatch, ClassBatch = next(minibatch_gen)
    # print("NumericalValueOfTrainSet :\n", NumericalValueOfTrainSet)
    # print("LinguisticBatch :\n", LinguisticBatch)
    # print("ClassValueOfTrainSet :\n", ClassValueOfTrainSet)

    loss, u_k, u_k_left, u_k_middle, u_k_right, u_left, u_middle, u_right, u_values = compute_loss_and_gradient(u_k, u_k_left, u_k_middle, u_k_right, u_left, u_middle, u_right, mOfBatch, NumericalBatch,
                                                                                                      LinguisticBatch, ClassBatch)

    u_ki1 = u_values["u_ki1"]
    u_kd1 = u_values["u_kd1"]
    epsilon_k1 = u_values["epsilon_k1"]
    u_k1 = u_values["u_k1"]
    u_k_left1 = u_values["u_k_left1"]
    u_k_middle1 = u_values["u_k_middle1"]
    u_k_right1 = u_values["u_k_right1"]
    u_left1 = u_values["u_left1"]
    u_middle1 = u_values["u_middle1"]
    u_right1 = u_values["u_right1"]

    loss_history.append((i, loss))
    parameter_history.append((i+1, u_ki1, u_kd1, epsilon_k1, u_k1, u_k_left1, u_k_middle1, u_k_right1,
                              u_left1, u_middle1, u_right1))

    MembershipMatrixOfTrainSet_history.append((i, MembershipMatrixOfTrainSet))
    CorrectOfTrainSet_history.append((i, CorrectOfTrainSet))
    SumCorrectOfTrainSet_history.append((i, SumCorrectOfTrainSet/mOfTrainSet))

    MembershipMatrixOfTestSet_history.append((i, MembershipMatrixOfTestSet))
    CorrectOfTestSet_history.append((i, CorrectOfTestSet))
    SumCorrectOfTestSet_history.append((i, SumCorrectOfTestSet/mOfTestSet))

    with open("parameter_history.pkl", "wb") as f:
        pickle.dump(parameter_history, f)
    with open("MembershipMatrixOfTrainSet_history.pkl", "wb") as f:
        pickle.dump(MembershipMatrixOfTrainSet_history, f)
    with open("MembershipMatrixOfTestSet_history.pkl", "wb") as f:
        pickle.dump(MembershipMatrixOfTestSet_history, f)

    np.save("loss_history.npy", loss_history)

    np.save("CorrectOfTrainSet_history.npy", CorrectOfTrainSet_history)
    np.save("SumCorrectOfTrainSet_history.npy", SumCorrectOfTrainSet_history)

    np.save("CorrectOfTestSet_history.npy", CorrectOfTestSet_history)
    np.save("SumCorrectOfTestSet_history.npy", SumCorrectOfTestSet_history)

    """
    if len(loss_history) > 1:
       
        previous_loss = loss_history[-2][1]

        StopCondition = abs(previous_loss - loss) / abs(previous_loss)
        StopCondition_history.append((i, StopCondition))
        print("StopCondition_history :\n", StopCondition_history)
        if StopCondition < tolerance:
            print(f"Relative change condition met at iteration {i}")
            break
    """




