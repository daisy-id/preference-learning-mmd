
Note: This repository is released under an Academic Research License. See LICENSE for details.

1.Project File Description
- DataSplit.py
  This handles splitting the data into the training and testing set.
- Preprocess.py
  This performs data preprocessing.
- InitialValue.py
  This sets the initial values for variables.
- Function.py
  This includes the preference and learning modules in the model.
- PartialDerivative
  This performs optimization to solve for model variables.
- Main
  This is the main script of the project, which serves as the entry point for model training and testing.
- Data/
  The data is used for experiment. Below are the descriptions of key variables:
  - Slope Structure
The encoding is as follows:
Code	Type
1	Near-horizontal layered slopes
2	Dip slopes
3	Anti-dip slopes
4	Lateral slopes
5	Inclined slopes
6	Downhill slopes

  - Stratigraphic Lithology
The encoding is as follows:
Code	Type
1	Hard and relatively hard karstified carbonate rocks
2	Hard thick-layered clastic rocks
3	Alternating soft and hard layered clastic rocks

  - Warning Level
The encoding is as follows:
Code	Type
4	Blue
3	Yellow
2	Orange
1	Red


2.Environment Setup
- Python version: 3.9
- OS: Windows 10 



