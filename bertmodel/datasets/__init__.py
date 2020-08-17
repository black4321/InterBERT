from .concept_cap_dataset import ConceptCapLoaderTrain, ConceptCapLoaderVal
from .retreival_dataset import RetreivalDataset, RetreivalDatasetVal
from .vcr_dataset import VCRDataset

__all__ = [
		   "ConceptCapLoaderTrain", \
		   "ConceptCapLoaderVal", \
		   "RetreivalDataset", \
		   "RetreivalDatasetVal",\
		   "VCRDataset"]

DatasetMapTrain = {
				   'TASK1': VCRDataset,
				   'TASK2': VCRDataset,				   
				   'TASK3': RetreivalDataset,
				   }		

DatasetMapEval = {
				 'TASK1': VCRDataset,
				 'TASK2': VCRDataset,				   
				 'TASK3': RetreivalDatasetVal,		   
				}
