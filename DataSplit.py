import pandas as pd
import os

# Set parameters
random_seed = 8
BiLiOfSample = 0.7  # 7:3

# Sample
samples_per_class_dict = {
    4: int(60 * BiLiOfSample),
    3: int(120 * BiLiOfSample),
    2: int(40 * BiLiOfSample),
    1: int(10 * BiLiOfSample)
}

# Reading Excel files
base_path = './Data'

SumSet_filename = 'Data.xlsx'
MatrixOfSumSet_template = os.path.join(base_path, SumSet_filename)
MatrixOfSumSet = pd.read_excel(MatrixOfSumSet_template)

# Obtain training and test sets
def undersampling_minibatch_generator(data, samples_per_class_dict, train_filename='TrainSet.xlsx', test_filename='TestSet.xlsx'):
    while True:

        fixed_train_set = data[data['Min-Max'] == 'min-max']
        remaining_data = data.drop(fixed_train_set.index)

        TrainSet_list = [fixed_train_set]
        TestSet_list = []
        for class_label, samples_count in samples_per_class_dict.items():
            class_data = remaining_data[remaining_data['Warning level'] == class_label]
            sampled_data = class_data.sample(samples_count, random_state=random_seed)
            remaining_data = remaining_data.drop(sampled_data.index)

            TrainSet_list.append(sampled_data)
            TestSet_list.append(remaining_data[remaining_data['Warning level'] == class_label])

        TrainSet = pd.concat(TrainSet_list).reset_index(drop=True)
        TestSet = pd.concat(TestSet_list).reset_index(drop=True)

        TrainSet = TrainSet.sort_values(by='Warning level', ascending=False).reset_index(drop=True)
        TestSet = TestSet.sort_values(by='Warning level', ascending=False).reset_index(drop=True)

        train_filepath = os.path.join(base_path, train_filename)
        test_filepath = os.path.join(base_path, test_filename)

        TrainSet.to_excel(train_filepath, index=False)
        TestSet.to_excel(test_filepath, index=False)

        yield TrainSet, TestSet

generator = undersampling_minibatch_generator(MatrixOfSumSet, samples_per_class_dict)
train_set, test_set = next(generator)
