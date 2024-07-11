# Information

The R-script was developed using RStudio, the Python-script was developed using Visual Studio Code. There might be additional steps needed to install tensorflow, when Visual Studio Code is used. By outcommenting all keras-packages that are imported at the top of the script, the script can still run until the neural network, but the latter requires tensorflow.

To run the first part of the code, open the RStudio Project "master_thesis". In the Project open "master_thesis_code_preparation_logistic_regression.R", this includes the first part of the code.
The second part of the code can be accessed by opening "master_thesis_code_machine_learning_methods.py".

Be aware, when runing the Python-script about 60 figures are plotted. Right now they are closed immediately, once the script ran, but there is an option at the top of the script to change this ("mode_figures"). However, changing this would lead to having to close about 60 tabs at the end. The fastest way to do so would be to close the python graphic window in the task bar.

Required and optional libraries/packages for R and Python can be found in the file "requirements_manual.txt".

The folder "models" contains pre-trained model-files, so the hyperparameter optimization does not have to be repeated everytime.

The folder "figures" contains graphs.

The folder "data" contains datasets.

The folder "outputs" contains information about prediction performances and temporary data that is used within the computations.