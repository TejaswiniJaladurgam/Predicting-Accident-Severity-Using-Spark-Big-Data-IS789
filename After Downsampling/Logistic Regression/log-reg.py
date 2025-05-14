import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics

spark=SparkSession.builder.appName("Accident_Severity_Logistic").getOrCreate()

#Load the dataset
df=spark.read.csv("/umbc/rs/is789sp25/users/xb56574/project/Accident_Information_Balanced.csv",inferSchema=True,header=True)

#Specify target and feature columns
target="Accident_Severity"
attributes=[col for col in df.columns if col != target]

#Using VectorAssembler
assembler=VectorAssembler(inputCols=attributes,outputCol="features")
df=assembler.transform(df)

#Split data: 70% for training and 30% for testing
train,test =df.randomSplit([0.7,0.3],seed=42)

#Logistic Regression Model
lr=LogisticRegression(featuresCol="features",labelCol=target)
model=lr.fit(train)
pred=model.transform(test)

#Evaluation
evaluator=MulticlassClassificationEvaluator(labelCol=target,predictionCol="prediction",metricName="accuracy")
acc=evaluator.evaluate(pred)
print("Test Accuracy:",acc)
pred_target=pred.select("prediction",target).rdd.map(lambda row:(float(row[0]),float(row[1])))
metrics=MulticlassMetrics(pred_target)
precision=metrics.weightedPrecision
recall=metrics.weightedRecall
f1=metrics.weightedFMeasure()
confusion_matrix = metrics.confusionMatrix().toArray()

np.set_printoptions(suppress=True) 

# Print evaluation metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:",)
print(confusion_matrix)

# Saving results into a text file
output_path = "/umbc/rs/is789sp25/users/xb56574/project/log-reg-bal-results.txt"
with open(output_path, "w") as f:
    f.write(f"Test Accuracy: {acc}\n")
    f.write(f"Test Error: {1.0 - acc}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Confusion Matrix:\n{confusion_matrix}\n")

spark.stop()


