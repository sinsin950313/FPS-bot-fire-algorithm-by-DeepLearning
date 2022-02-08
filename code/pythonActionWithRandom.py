import fireDecision
fireDecision = fireDecision.fireDecision()

import trainingDataSet
trainingDataSet = trainingDataSet.trainingDataSet()
trainingDataSet.create_Datas(0.8, 0.6)

import variableDataSet
variableDataSet = variableDataSet.variableDataSet()
variableDataSet.create_Datas()

import trainer
trainer = trainer.trainer()

trainer.training(fireDecision, trainingDataSet, 8000)
