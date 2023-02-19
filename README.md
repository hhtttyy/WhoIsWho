# WhoIsWho Toolkit
**TL;DR**: By automating data loading, feature creation, model construction, and evaluation processes, the WhoIsWho toolkit is easy for researchers to use and let them develop new name disambiguation approaches.

The toolkit is fully compatible with PyTorch and its associated deep-learning libraries, such as Hugging face. Additionally, the toolkit offers library-agnostic dataset objects that can be used by any other Python deep-learning framework such as TensorFlow.  

## Overview

<img src="whoiswho_pipeline.png" alt="shot" style="zoom:50%;" />

## Install

```
pip install whoiswho
```


## Pipeline
The WhoIsWho toolkit aims at providing lightweight APIs to facilitate researchers to build SOTA name disambiguation algorithms with several lines of code. The abstraction has 4 parts:
* **WhoIsWho Data Loader**: Automating dataset processing and splitting. 
* **Feature Creation**: Providing flexible modules for extracting and creating features.
* **Model Construction**: Adopting pre-defined models in the toolkit library for training and prediction.
* **WhoIsWho Evaluator**: Evaluating models in a task-dependent manner and output the model performance on the validation set.


### Commands
Todo...

### Examples of Build Basic RND Algotithms.

```python
# Module-1: Data Loading
from whoiswho.dataset import LoadData, SplitDataRND
# Load specific versions of dataset.
train = LoadData(name="v3", type="train", partition=None)
# Split data into unassigned papers and candidate authors
unassigns, candidates = SplitDataRND(train, split="time", ratio=0.2)

# Modules-2: Feature Creation
from whoiswho.featureGenerator import AdHocFeatures
# Extract default n-dimensional ad-hoc features.
pos_feats, neg_feats = AdHocFeatures(unassigns, candidates, feature_mode="default", negatives=3)

# Module-3: Model Construction
from whoiswho.loadmodel import ClassficationModels
# build a basic classfication model.
predictor=ClassficationModels(type="MLP", ensemble=False)
# Automatic training 
from whoiswho.training import AutoTrainRND
predictor = AutoTrainRND(inputs = (pos_feats, neg_feats), predictor, epoch=1, bs=1, early_stop=None)

# Modules-4: Evaluation on the validation data
# Load validation data
unassigns, candidates, gt = LoadData("v3", type="Valid", task="RND")
# Assign unassigned papers
assign_res = predictor.predict(unassigns, candidates)
# Evaluate the RND results
from whoiswho.evaluation import RNDeval
weighted_Precision, weighted_Recall, weighted_F1 = RNDeval(assign_res, gt)
```
