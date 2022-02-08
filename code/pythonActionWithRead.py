import fireDecision
fireDecision = fireDecision.fireDecision()

import trainingDataSet
trainingDataSet = trainingDataSet.trainingDataSet()
trainingDataSet.read_Datas()

import trainer
trainer = trainer.trainer()

trainer.training(fireDecision, trainingDataSet, 8000)
