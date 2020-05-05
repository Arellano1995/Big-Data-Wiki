import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Load the data stored in LIBSVM format as a DataFrame.
// Cargamos los datos desde el archivo origen
val data = spark.read.format("libsvm").load("Big-Data-Wiki/Example/sample_libsvm_data.txt")
data.show()


// StringIndexer 
// convierte una sola columna en una columna de índice (similar a una columna de factor en R)
// Si la columna de entrada es numérica, la convertimos en una cadena e indexamos los valores de la cadena.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

// VectorIndexer
// Clase para indexar columnas de entidades categóricas en un conjunto de datos de Vector.
// se usa para indexar predictores categóricos en una columna featuresCol. 
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)


// Dividimos de forma aleateoria los datos en 70% y 30%, para entrenamiento y prueba, respectivamente.
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))


// Algoritmo de aprendizaje del árbol de decisión (Entrenamos el algoritmo).
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

// Revertimos la conversion, de etiquetas indexadas a etiquetas originales.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// ML Pipelines proporciona un conjunto uniforme de API de alto nivel creadas sobre DataFrames 
// que ayudan a los usuarios a crear y ajustar tuberías prácticas de aprendizaje automático.
// https://spark.apache.org/docs/latest/ml-pipeline.html
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// Utilizamos el modelo de entrenamiento, con los datos de entrenamiento (70%)
val model = pipeline.fit(trainingData)

// Realizamos predicciones con los datos de prueba (30%)
val predictions = model.transform(testData)

// Mostramos las filas de ejemplo
predictions.select("predictedLabel", "label", "features").show(5)

// Evaluador para clasificación multiclase, que espera dos columnas de entrada: predicción y etiqueta.
// Calculamos el error de prueba
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

//Instanciamos el modelo aprendido
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")