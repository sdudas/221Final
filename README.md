# 221Final


## Setup
Follow these steps for installation.

### Manual Installation
Clone this repository.

Then change your directory to the Project directory.

Currently, this package works with the sensitive attribute race.

The below example is to check the fairness of the model for race attribute.

Steps:

`git clone git@github.com:sdudas/221Final.git`

`cd FairClassifier`

`python src/notebooks/main.py race African-American`

## Architecture
The overview of the architecture of the model is shown in this figure below:

![](images/architecture.png)

The system of training two Neural Network might look similar to the one used for training [GANS.](https://arxiv.org/abs/1406.2661) 
However, it is not the case. 

First, the generative model used in GANs is replaced by a predictive model which 
instead of generating synthetic data gives actual predictions from the input data.
Second, adversarial network doesn't distinguish real data from generated synthetic data in this case. 
However, it tries to predict the sensitive attribute values from the predicted labels from the earlier Neural Network.
Both of these networks train simultaneously with an objective to optimize the prediction losses of prediction labels and sensitive attributes. 

### DataSet
The [Dataset](https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv) used in this project 
was acquired, analyzed and released by [Propublica.](https://github.com/propublica/compas-analysis) It consists of ~12k records of criminals of Broward County. 

Using this dataset, the model predicts how likely a criminal defendant is to reoffend.
Recidivism is defined as a new arrest within two years in the analysis of data by [Propublica.](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm)
Each defendant received a score for 'Risk of recidivism' also called as COMPAS score. 
The score for each defendant ranged from 1 to 10. 
To start with a binary classification problem, Scores 1 to 5 were re-labeled as 'Low'
and 6-10 were re-labeled as 'High'. 
 
Some of the important attributes associated with each criminal defendants are:

* Sex
* Age 
* Race 
* Prior Arrest Count
* Days arrested before assessment
* Score Label
