import fireDecision
fireDecision = fireDecision.fireDecision()

import trainingDataSet
trainingDataSet = trainingDataSet.trainingDataSet()
trainingDataSet.read_Datas()

import variableDataSet
variableDataSet = variableDataSet.variableDataSet()

import trainer
trainer = trainer.trainer()

trainer.training(fireDecision, trainingDataSet, 50)
