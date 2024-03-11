This code reproduces the experimental results from the experimental study in Section 6.

# Installation
1. Clone the repository to your local directory.
2. Install the requirements: `pip install -r requirements.txt`.


# Execution 
The script `RunExperiment.py` reproduces a single experiment. For example, in order to reproduce the USPS to MNIST experiment for 3 samples per target class, use the following:
````
python RunExperiment.py --Src "U" --Tgt "M" --SamplesPerClass 3 --Method "SDA_IO" --GPU_ID -1
````
Type `python RunExperiment.py -h` to view the argument's documentation.
More examples for executing this script from the command line are available in `Examples.txt`.
A single experiment lasts approximately 2 hours when using a CPU, when using an NVIDIA Tesla V100 GPU it is reduced to 30 minutes.

In order to reproduce all the experiments use `RunAllExperiments.bat` for Windows OS and `RunAllExperiments.sh` for Linux OS.
We recommend monitoring the results using W&B - `https://wandb.ai` (requires an account). We support this ability by setting the `LogToWandb` flag to True:
````
python RunExperiment.py --Src "U" --Tgt "M" --SamplesPerClass 3 --Method "SDA_IO" --GPU_ID -1 --LogToWandb True
```` 
If set to True, then it is required to modify the username accordingly in line 14 in `Utilities\Configuration_Utils.py`.


# Changing the hyperparameters
The hyperparameters are listed in `Utilities\Configuration_Utils.py` in the function `GetConfFromArgs`.
The hyperparameters are set according to the experiment regardless of the evaluated method (Ours,CCSA,dSNE,NEM)

# Credits
The data augmentation and the code for the backbone in the ''Office" experiment were implemented using the 
Transfer Learning Library: `https://github.com/thuml/Transfer-Learning-Library/`.