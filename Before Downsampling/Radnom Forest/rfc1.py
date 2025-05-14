from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.appName("AccidentSeverityPrediction").getOrCreate()

df=spark.read.csv("/umbc/rs/is789sp25/users/xb56574/project/Accident_Information_preprocessed.csv", inferSchema=True, header=True)
df.show()

target=df.columns[3]
attributes=df.columns[:3]+df.columns[4:]

assembler=VectorAssembler(inputCols=attributes, outputCol="attributes")
df=assembler.transform(df)

train, test= df.randomSplit([0.7, 0.3], seed=42)
train.printSchema()

rfc=RandomForestClassifier(labelCol=target, featuresCol="attributes", numTrees=20, seed=42)
model=rfc.fit(train)

pred=model.transform(test)

# Evaluation
evaluator=MulticlassClassificationEvaluator(labelCol=target, predictionCol="prediction", metricName="accuracy")
acc=evaluator.evaluate(pred)
print("Test Accuracy: ", acc)

pred_target=pred.select("prediction", target).rdd.map(lambda row: (float(row[0]), float(row[1])))
metrics=MulticlassMetrics(pred_target)
precision=metrics.weightedPrecision
recall=metrics.weightedRecall
f1=metrics.weightedFMeasure()
confusion_matrix=metrics.confusionMatrix().toArray()

print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)
print("Confusion Matrix:\n", confusion_matrix)

#Saving results into a text file
with open("rfc1-results.txt", "w") as f:
        f.write(f"Test Accuracy: {acc}\n")
        f.write(f"Test Error: {1.0-acc}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Confusion Matrix:\n {confusion_matrix}\n")
        f.write(f"Feature Importances: {model.featureImportances}\n")

spark.stop()

