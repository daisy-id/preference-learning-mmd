import torch
import numpy as np
from Function import membership_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def alternative_membership_u_k(x, u_left, u_middle, u_right, AlternativeU_k, LinguisticValueSumUtilityLeft, LinguisticValueSumUtilityMiddle, LinguisticValueSumUtilityRight):
    x = np.array(x, dtype=np.float32)
    membership = np.zeros_like(x, dtype=np.float32)

    if u_left == u_middle == u_right:
        return membership
    elif u_left == u_middle:
        mask = (u_middle <= x) & (x < u_right)

        return np.where(mask, AlternativeU_k / (LinguisticValueSumUtilityRight - LinguisticValueSumUtilityMiddle), 0.0)
    elif u_middle == u_right:
        mask = (u_left <= x) & (x < u_middle)
        return np.where(mask, - AlternativeU_k / (LinguisticValueSumUtilityMiddle - LinguisticValueSumUtilityLeft), 0.0)
    else:
        mask1 = (u_left <= x) & (x < u_middle)
        membership[mask1] = - AlternativeU_k / (LinguisticValueSumUtilityMiddle - LinguisticValueSumUtilityLeft)
        mask2 = (u_middle <= x) & (x < u_right)
        membership[mask2] = AlternativeU_k / (LinguisticValueSumUtilityRight - LinguisticValueSumUtilityMiddle)
        return membership

def membership_u_k_middle(x, u_left, u_middle, u_right, LinguisticValueSumUtilityLeft, LinguisticValueSumUtilityMiddle, LinguisticValueSumUtilityRight):
    x = np.array(x, dtype=np.float32)

    if u_left == u_middle == u_right:
        return np.zeros_like(x, dtype=np.float32)
    elif u_left == u_middle:
        mask = (u_middle <= x) & (x < u_right)
        return np.where(mask, (u_right-x)/((LinguisticValueSumUtilityRight-LinguisticValueSumUtilityMiddle)**2), 0.0)
    elif u_middle == u_right:
        mask = (u_left <= x) & (x < u_middle)
        return np.where(mask, (u_left-x)/((LinguisticValueSumUtilityMiddle-LinguisticValueSumUtilityLeft)**2), 0.0)
    else:
        membership = np.zeros_like(x, dtype=np.float32)
        mask1 = (u_left <= x) & (x < u_middle)
        membership[mask1] = (u_left-x[mask1])/((LinguisticValueSumUtilityMiddle-LinguisticValueSumUtilityLeft)**2)
        mask2 = (u_middle <= x) & (x < u_right)
        membership[mask2] = (u_right-x[mask2])/((LinguisticValueSumUtilityRight-LinguisticValueSumUtilityMiddle)**2)
        return membership

def alternative_loss_u_k_computer(params, AlternativeU_k, LinguisticValueSumUtilityLeft, LinguisticValueSumUtilityMiddle, LinguisticValueSumUtilityRight):
    alternative_left, alternative_middle, alternative_right, class_left, class_middle, class_right = params
    points = sorted([alternative_left, alternative_middle, alternative_right, class_left, class_middle, class_right])
    total_discuss = 0.0
    for p in range(len(points) - 1):
        x_i, x_next = points[p], points[p + 1]
        x = np.linspace(x_i, x_next, 100)
        y = membership_function(x, class_left, class_middle, class_right)*alternative_membership_u_k(x, alternative_left, alternative_middle, alternative_right, AlternativeU_k, LinguisticValueSumUtilityLeft, LinguisticValueSumUtilityMiddle, LinguisticValueSumUtilityRight)
        integral = np.trapz(y, x)

        membership_x_i = membership_function(x_i, class_left, class_middle, class_right) * membership_function(x_i, alternative_left, alternative_middle, alternative_right)
        membership_x_next = membership_function(x_next, class_left, class_middle, class_right) * membership_function(x_next, alternative_left, alternative_middle, alternative_right)

        alternative_points = np.array([alternative_left, alternative_middle, alternative_right])
        x_i_u_k = np.where(np.isin(x_i, alternative_points), AlternativeU_k, 0)
        x_next_u_k = np.where(np.isin(x_next, alternative_points), AlternativeU_k, 0)
        # print(f"x_i: {x_i}, x_i_u_k: {x_i_u_k}")
        # print(f"x_next: {x_next}, x_next_u_k: {x_next_u_k}")

        discuss = membership_x_next * x_next_u_k - membership_x_i * x_i_u_k + integral
        total_discuss += discuss
    AlternativeLossU_k = 1/(LinguisticValueSumUtilityRight-LinguisticValueSumUtilityLeft)*total_discuss
    return AlternativeLossU_k

def alternative_loss_u_k(u_k, NumericalValue, low, upper, params, LinguisticValueSumUtilityLeft, LinguisticValueSumUtilityMiddle, LinguisticValueSumUtilityRight):
    AlternativeU_k = np.zeros_like(u_k, dtype=np.float32)
    AlternativeLossU_k = np.zeros_like(u_k, dtype=np.float32)

    # print(f"u_k: {u_k}, low: {low}, upper: {upper}")
    low_mask = (u_k == low)
    upper_mask = (u_k == upper)

    AlternativeU_k[low_mask] = 1 - (NumericalValue - low) / (upper - low)
    AlternativeU_k[upper_mask] = (NumericalValue - low) / (upper - low)

    AlternativeLossU_k[low_mask] = alternative_loss_u_k_computer(params, AlternativeU_k, LinguisticValueSumUtilityLeft, LinguisticValueSumUtilityMiddle, LinguisticValueSumUtilityRight)
    AlternativeLossU_k[upper_mask] = alternative_loss_u_k_computer(params, AlternativeU_k, LinguisticValueSumUtilityLeft, LinguisticValueSumUtilityMiddle,  LinguisticValueSumUtilityRight)
    return AlternativeLossU_k



def alternative_loss_u_k_middle_computer(params, LinguisticValueSumUtilityLeft, LinguisticValueSumUtilityMiddle, LinguisticValueSumUtilityRight):
    alternative_left, alternative_middle, alternative_right, class_left, class_middle, class_right = params
    points = sorted([alternative_left, alternative_middle, alternative_right, class_left, class_middle, class_right])
    total_discuss = 0.0
    for p in range(len(points) - 1):
        x_i, x_next = points[p], points[p + 1]
        x = np.linspace(x_i, x_next, 100)
        y = membership_function(x, class_left, class_middle, class_right)*membership_u_k_middle(x, alternative_left, alternative_middle, alternative_right, LinguisticValueSumUtilityLeft, LinguisticValueSumUtilityMiddle, LinguisticValueSumUtilityRight)
        integral = np.trapz(y, x)

        membership_x_i = membership_function(x_i, class_left, class_middle, class_right) * membership_function(x_i, alternative_left, alternative_middle, alternative_right)
        membership_x_next = membership_function(x_next, class_left, class_middle, class_right) * membership_function(x_next, alternative_left, alternative_middle, alternative_right)

        x_i_u_k_middle = np.where(x_i == alternative_middle, 1, 0)
        x_next_u_k_middle = np.where(x_next == alternative_middle, 1, 0)

        discuss = membership_x_next * x_next_u_k_middle - membership_x_i * x_i_u_k_middle + integral
        total_discuss += discuss
    AlternativeLossU_k_middle = 1/(LinguisticValueSumUtilityRight-LinguisticValueSumUtilityLeft)*total_discuss
    return AlternativeLossU_k_middle

def alternative_loss_u_k_middle(u_k_middle, LinguisticValueUtilityMiddle, params, LinguisticValueSumUtilityLeft, LinguisticValueSumUtilityMiddle, LinguisticValueSumUtilityRight):
    AlternativeLossU_k_middle = np.zeros_like(u_k_middle, dtype=np.float32)

    mask = (u_k_middle == LinguisticValueUtilityMiddle)
    AlternativeLossU_k_middle[mask] = alternative_loss_u_k_middle_computer(params, LinguisticValueSumUtilityLeft, LinguisticValueSumUtilityMiddle, LinguisticValueSumUtilityRight)
    return AlternativeLossU_k_middle


def alternative_loss_u_middle_computer(params, LinguisticValueSumUtilityLeft, LinguisticValueSumUtilityRight):
    alternative_left, alternative_middle, alternative_right, class_left, class_middle, class_right = params
    points = sorted([alternative_left, alternative_middle, alternative_right, class_left, class_middle, class_right])
    total_discuss = 0.0
    for p in range(len(points) - 1):
        x_i, x_next = points[p], points[p + 1]
        x = np.linspace(x_i, x_next, 100)
        y = membership_function(x, alternative_left, alternative_middle, alternative_right)*membership_u_k_middle(x, class_left, class_middle, class_right, class_left, class_middle, class_right)
        integral = np.trapz(y, x)

        membership_x_i = membership_function(x_i, class_left, class_middle, class_right) * membership_function(x_i, alternative_left, alternative_middle, alternative_right)
        membership_x_next = membership_function(x_next, class_left, class_middle, class_right) * membership_function(x_next, alternative_left, alternative_middle, alternative_right)

        x_i_u_middle = np.where(x_i == class_middle, 1, 0)
        x_next_u_middle = np.where(x_next == class_middle, 1, 0)

        discuss = membership_x_next * x_next_u_middle - membership_x_i * x_i_u_middle + integral
        total_discuss += discuss
    AlternativeLossU_middle = 1/(LinguisticValueSumUtilityRight-LinguisticValueSumUtilityLeft)*total_discuss
    return AlternativeLossU_middle


def alternative_loss_u_middle(u_middle, ClassValueUtilityMiddle, params, LinguisticValueSumUtilityLeft, LinguisticValueSumUtilityRight):
    AlternativeLossU_middle = np.zeros_like(u_middle, dtype=np.float32)

    mask = (u_middle == ClassValueUtilityMiddle)
    AlternativeLossU_middle[mask] = alternative_loss_u_middle_computer(params, LinguisticValueSumUtilityLeft, LinguisticValueSumUtilityRight)
    return AlternativeLossU_middle






