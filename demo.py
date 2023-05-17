from whoiswho.dataset import LoadData,processdata_RND,processdata_SND
from whoiswho.featureGenerator.rndFeature import AdhocFeatures , OagbertFeatures
from whoiswho.training import RNDTrainer
from whoiswho.evaluation import evaluate
from whoiswho.utils import load_json
from whoiswho.config  import *
from whoiswho import logger

'''
    RND task
'''
# Module-1: Data Loading
train,version = LoadData(name="v3", type="train",task='RND')
valid,version = LoadData(name="v3", type="valid",task='RND')
test,version  = LoadData(name="v3", type="test", task='RND')

# Split data into unassigned papers and candidate authors
# Combine unassigned papers and candidate authors into train pairs.
train, version = LoadData(name="v3", type="train", task='RND', download=False)
processdata_RND(train,version)
logger.info("Finish pre-process")

# Modules-2: Feature Creation
data, version = LoadData(name="v3", type="train", task='RND', download=False)
adhoc_features = AdhocFeatures(version)
adhoc_features.get_hand_feature()
oagbert_features = OagbertFeatures(version)
oagbert_features.get_oagbert_feature()
logger.info("Finish Train data")

data, version = LoadData(name="v3", type="valid", task='RND', download=False)
adhoc_features = AdhocFeatures(version)
adhoc_features.get_hand_feature()
oagbert_features = OagbertFeatures(version)
oagbert_features.get_oagbert_feature()
logger.info("Finish Valid data")

data, version = LoadData(name="v3", type="test", task='RND', download=False)
adhoc_features = AdhocFeatures(version)
adhoc_features.get_hand_feature()
oagbert_features = OagbertFeatures(version)
oagbert_features.get_oagbert_feature()
logger.info("Finish Test data")

# Module-3: Model Construction
data, version = LoadData(name="v3", type="train", task='RND', download=False)
trainer = RNDTrainer(version)
cell_model_list = trainer.fit()
logger.info("Finish Training")

data, version = LoadData(name="v3", type="valid", task='RND', download=False)
trainer = RNDTrainer(version)
trainer.predict(cell_model_list=cell_model_list)
logger.info("Finish Predict Valid data")

data, version = LoadData(name="v3", type="test", task='RND', download=False)
trainer = RNDTrainer(version)
trainer.predict(cell_model_list=cell_model_list)
logger.info("Finish Predict Test data")

# Modules-4: Evaluation on the valid data
assignment = load_json('./whoiswho/training/result/result.valid.json')
#Use the ground truth directly or load the downloaded ground truth
ground_truth = valid[4]
evaluate(assignment, ground_truth)
