import numpy as np
import pandas as pd
import os
from Function import numerical_value_breakpoint_matrix, linguistic_value_breakpoint_matrix

# Set parameters
GammaOfNumericalValue = [4, 5, 6, 7, 8]
GammaOfLinguisticValue = [6, 3]

# Reading Excel files
base_path = './Data'

TrainSet_filename = 'TrainSet.xlsx'
TestSet_filename = 'TestSet.xlsx'
RangeSet_filename = 'RangeSet.xlsx'

MatrixOfTrainSet_template = os.path.join(base_path, TrainSet_filename)
MatrixOfTestSet_template = os.path.join(base_path, TestSet_filename)
MatrixOfRangeSet_template = os.path.join(base_path, RangeSet_filename)

MatrixOfTrainSet = pd.read_excel(MatrixOfTrainSet_template)
MatrixOfTestSet = pd.read_excel(MatrixOfTestSet_template)
MatrixOfRangeSet = pd.read_excel(MatrixOfRangeSet_template)
print(f"MatrixOfRangeSet: {MatrixOfRangeSet}")

MatrixOfSumSet = pd.concat([MatrixOfTrainSet, MatrixOfTestSet], ignore_index=True)

def normalize(matrix, MatrixOfSumSet,  normalize_first_n_cols=10):

    data_to_normalize = matrix[:, :normalize_first_n_cols].astype(np.float64)
    data_to_keep = matrix[:, normalize_first_n_cols:]

    # min, max

    sum_set_values = MatrixOfSumSet.iloc[:, :normalize_first_n_cols].values.astype(np.float64)
    min_vals = np.min(sum_set_values, axis=0)
    max_vals = np.max(sum_set_values, axis=0)

    normalized_data = (data_to_normalize - min_vals) / (max_vals - min_vals)

    normalized_matrix = np.hstack([normalized_data, data_to_keep])

    return normalized_matrix

MatrixOfTrainSet_normalized = normalize(MatrixOfTrainSet.values, MatrixOfRangeSet)
MatrixOfTestSet_normalized = normalize(MatrixOfTestSet.values, MatrixOfRangeSet)
MatrixOfRangeSet_normalized = normalize(MatrixOfRangeSet.values, MatrixOfRangeSet)

MatrixOfTrainSet_normalized_df = pd.DataFrame(MatrixOfTrainSet_normalized, columns=MatrixOfTrainSet.columns)
MatrixOfTestSet_normalized_df = pd.DataFrame(MatrixOfTestSet_normalized, columns=MatrixOfTestSet.columns)
MatrixOfRangeSet_normalized_df = pd.DataFrame(MatrixOfRangeSet_normalized, columns=MatrixOfRangeSet.columns)

train_set_normalized_filename = 'TrainSet_normalized.xlsx'
test_set_normalized_filename = 'TestSet_normalized.xlsx'
range_set_normalized_filename = 'RangeSet_normalized.xlsx'

train_set_normalized_path = os.path.join(base_path, train_set_normalized_filename)
test_set_normalized_path = os.path.join(base_path, test_set_normalized_filename)
range_set_normalized_path = os.path.join(base_path, range_set_normalized_filename)

MatrixOfTrainSet_normalized_df.to_excel(train_set_normalized_path, index=False)
MatrixOfTestSet_normalized_df.to_excel(test_set_normalized_path, index=False)
MatrixOfRangeSet_normalized_df.to_excel(range_set_normalized_path, index=False)


NumericalValueOfRangeSet = MatrixOfRangeSet.iloc[:, :8].values
LinguisticValueOfRangeSet = MatrixOfRangeSet.iloc[:, 8:10].values
nOfNumericalValue = NumericalValueOfRangeSet .shape[1]
nOfLinguisticValue = LinguisticValueOfRangeSet .shape[1]

NumericalValueBreakpoints = []
for Gamma in GammaOfNumericalValue:
    NumericalValueBreakpoint = numerical_value_breakpoint_matrix(nOfNumericalValue, NumericalValueOfRangeSet, Gamma)

    NumericalValueBreakpoints.append(NumericalValueBreakpoint)

LinguisticValueBreakpoint = linguistic_value_breakpoint_matrix(nOfLinguisticValue, LinguisticValueOfRangeSet, GammaOfLinguisticValue)

max_cols_numerical = max([arr.shape[1] for arr in NumericalValueBreakpoints])

NumericalValueBreakpoints_padded = [
    np.pad(arr, ((0, 0), (0, max_cols_numerical - arr.shape[1])), mode='constant', constant_values=np.nan)
    for arr in NumericalValueBreakpoints
]

NumericalValueBreakpoint_df = pd.DataFrame(np.vstack(NumericalValueBreakpoints_padded))
LinguisticValueBreakpoint_df = pd.DataFrame(LinguisticValueBreakpoint)

Breakpoint_filename = 'Breakpoint.xlsx'
file_path = os.path.join(base_path, Breakpoint_filename)

with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
    NumericalValueBreakpoint_df.to_excel(writer, sheet_name='NumericalValueBreakpoint', index=False)
    LinguisticValueBreakpoint_df.to_excel(writer, sheet_name='LinguisticValueBreakpoint', index=False)

print(f"Breakpoint saved to: {file_path}")
