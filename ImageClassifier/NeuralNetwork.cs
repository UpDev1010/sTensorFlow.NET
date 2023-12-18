using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace ImageClassifier
{
    public class NeuralNetwork
    {
        Matrix<double> weightsInputHidden;
        Matrix<double> biasesHidden;
        Matrix<double> weightsOutputHidden;
        Matrix<double> biasesOutput;

        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
        {
            weightsInputHidden = Matrix<double>.Build.Random(inputSize, hiddenSize);
            biasesHidden = Matrix<double>.Build.Random(1, hiddenSize);
            weightsOutputHidden = Matrix<double>.Build.Random(hiddenSize, outputSize);
            biasesOutput = Matrix<double>.Build.Random(1, outputSize);
        }

        public void Save(string folderPath)
        {
            //Newtonsoft.Json.JsonSerializer serializer = new Newtonsoft.Json.JsonSerializer();
            //serializer.Converters.Add(new Newtonsoft.Json.Converters.JavaScriptDateTimeConverter());
            //serializer.NullValueHandling = Newtonsoft.Json.NullValueHandling.Ignore;
            //serializer.TypeNameHandling = Newtonsoft.Json.TypeNameHandling.Auto;
            //serializer.Formatting = Newtonsoft.Json.Formatting.Indented;

            Directory.CreateDirectory(folderPath);
            string wIHPath = Path.Combine(folderPath, "wIH.bin");
            string bIHPath = Path.Combine(folderPath, "bIH.bin");
            string wOHPath = Path.Combine(folderPath, "wOH.bin");
            string bOHPath = Path.Combine(folderPath, "bOH.bin");

            if (File.Exists(wIHPath)) File.Delete(wIHPath);
            if (File.Exists(bIHPath)) File.Delete(bIHPath);
            if (File.Exists(wOHPath)) File.Delete(wOHPath);
            if (File.Exists(bOHPath)) File.Delete(bOHPath);

            //using (StreamWriter sw = new StreamWriter(wIHPath))
            //using (Newtonsoft.Json.JsonWriter writer = new Newtonsoft.Json.JsonTextWriter(sw))
            //{
            //    serializer.Serialize(writer, weightsInputHidden, typeof(Matrix<double>));
            //}
            //using (StreamWriter sw = new StreamWriter(bIHPath))
            //using (Newtonsoft.Json.JsonWriter writer = new Newtonsoft.Json.JsonTextWriter(sw))
            //{
            //    serializer.Serialize(writer, biasesHidden, typeof(Matrix<double>));
            //}
            //using (StreamWriter sw = new StreamWriter(wOHPath))
            //using (Newtonsoft.Json.JsonWriter writer = new Newtonsoft.Json.JsonTextWriter(sw))
            //{
            //    serializer.Serialize(writer, weightsOutputHidden, typeof(Matrix<double>));
            //}
            //using (StreamWriter sw = new StreamWriter(bOHPath))
            //using (Newtonsoft.Json.JsonWriter writer = new Newtonsoft.Json.JsonTextWriter(sw))
            //{
            //    serializer.Serialize(writer, biasesOutput, typeof(Matrix<double>));
            //}

            File.WriteAllText(wIHPath, JsonConvert.SerializeObject(weightsInputHidden.ToArray()));
            File.WriteAllText(bIHPath, JsonConvert.SerializeObject(biasesHidden.ToArray()));
            File.WriteAllText(wOHPath, JsonConvert.SerializeObject(weightsOutputHidden.ToArray()));
            File.WriteAllText(bOHPath, JsonConvert.SerializeObject(biasesOutput.ToArray()));
        }

        public bool Load(string folderPath)
        {
            if (!Directory.Exists(folderPath))
                return false;
            var settings = new JsonSerializerSettings { Converters = new List<JsonConverter> { new MatrixJsonConverter<double>() } };

            string wIHPath = Path.Combine(folderPath, "wIH.bin");
            string bIHPath = Path.Combine(folderPath, "bIH.bin");
            string wOHPath = Path.Combine(folderPath, "wOH.bin");
            string bOHPath = Path.Combine(folderPath, "bOH.bin");

            if (!File.Exists(bIHPath)) return false;
            if (!File.Exists(wOHPath)) return false;
            if (!File.Exists(bOHPath)) return false;
            if (!File.Exists(wIHPath)) return false;

            weightsInputHidden = JsonConvert.DeserializeObject<Matrix<double>>(File.ReadAllText(wIHPath), settings);
            biasesHidden = JsonConvert.DeserializeObject<Matrix<double>>(File.ReadAllText(bIHPath), settings);
            weightsOutputHidden = JsonConvert.DeserializeObject<Matrix<double>>(File.ReadAllText(wOHPath), settings);
            biasesOutput = JsonConvert.DeserializeObject<Matrix<double>>(File.ReadAllText(bOHPath), settings);

            return true;
        }

        public double Train(Matrix<double> xBatch, Matrix<double> yBatchNonOneHot, double learningRate)
        {
            var hiddenLayerInput = xBatch * weightsInputHidden + biasesHidden;
            var hiddenLayerOutput = Sigmoid(hiddenLayerInput);
            var outputLayerInput = hiddenLayerOutput * weightsOutputHidden + biasesOutput;
            var outputLayerOutput = Softmax(outputLayerInput);

            var yBatch = OneHotEncoding(yBatchNonOneHot, outputLayerInput.ColumnCount);

            // backward propagation
            var loss = CrossEntropyLoss(outputLayerOutput, yBatch);

            var outputDelta = outputLayerOutput - yBatch; // a2 - y

            var tt = (outputDelta * weightsOutputHidden.Transpose());
            var hiddenDelta = tt.PointwiseMultiply(hiddenLayerOutput).PointwiseMultiply(1 - hiddenLayerOutput);

            weightsOutputHidden -= hiddenLayerOutput.Transpose() * outputDelta * learningRate;
            biasesOutput -= outputDelta * learningRate;
            weightsInputHidden -= xBatch.Transpose() * hiddenDelta * learningRate;
            biasesHidden -= hiddenDelta * learningRate;

            return loss;
        }

        public Matrix<double> OneHotEncoding(Matrix<double> x, int newRowSize)
        {
            var retMat = Matrix<double>.Build.Dense(newRowSize, 1);
            retMat[(int)x[0, 0], 0] = 1;
            return retMat.Transpose();
        }

        public double Evaluate(Matrix<double> xTest, Matrix<double> yTest)
        {
            var predictions = Predict(xTest);
            var correct = 0;
            for (int i = 0; i < yTest.RowCount; i++)
            {
                int actual = Convert.ToInt32(yTest[i, 0]);
                var predicted = predictions.Row(i).MaximumIndex();

                if (actual == predicted)
                    correct++;
            }

            return (double)correct / yTest.RowCount;
        }

        public Matrix<double> Predict(Matrix<double> xTest)
        {
            var extendedBiasesHidden = Matrix<double>.Build.DenseOfRowVectors(Enumerable.Repeat(biasesHidden.Row(0), xTest.RowCount));

            var hiddenLayerInput = (xTest * weightsInputHidden) + extendedBiasesHidden;
            var hiddenLayerOutput = Sigmoid(hiddenLayerInput);

            var extendedBiasesOutput = Matrix<double>.Build.DenseOfRowVectors(Enumerable.Repeat(biasesOutput.Row(0), xTest.RowCount));

            var outputLayerInput = hiddenLayerOutput * weightsOutputHidden + extendedBiasesOutput;
            var outputLayerOutput = Softmax(outputLayerInput);
            return outputLayerOutput;
        }

        public int PredictNumber(Matrix<double> xTest)
        {
            var extendedBiasesHidden = Matrix<double>.Build.DenseOfRowVectors(Enumerable.Repeat(biasesHidden.Row(0), xTest.RowCount));

            var hiddenLayerInput = (xTest * weightsInputHidden) + extendedBiasesHidden;
            var hiddenLayerOutput = Sigmoid(hiddenLayerInput);

            var extendedBiasesOutput = Matrix<double>.Build.DenseOfRowVectors(Enumerable.Repeat(biasesOutput.Row(0), xTest.RowCount));

            var outputLayerInput = hiddenLayerOutput * weightsOutputHidden + extendedBiasesOutput;
            return outputLayerInput.Row(0).MaximumIndex();
        }

        private double CrossEntropyLoss(Matrix<double> predicted, Matrix<double> actual)
        {
            int numRows = predicted.RowCount;
            int numClasses = predicted.ColumnCount;

            // Calculate the cross-entropy loss
            double loss = 0.0;
            for (int i = 0; i < numRows; i++)
            {
                for (int j = 0; j < numClasses; j++)
                {
                    loss -= actual[i, j] * Math.Log(predicted[i, j] + 1e-15); // Add a small epsilon to prevent log(0)
                }
            }

            // Normalize the loss by the number of samples
            loss /= numRows;

            return loss;
        }

        private Matrix<double> Sigmoid(Matrix<double> x)
        {
            return 1 / (1 + (-x).PointwiseExp());
        }

        private Matrix<double> Exp(Matrix<double> x)
        {
            return x.Map(Math.Exp);
        }

        private Matrix<double> Softmax(Matrix<double> x)
        {
            // Exponentiate the logits
            var expLogits = x.PointwiseExp();

            // Calculate the sum of the exponentiated logits
            var sumExpLogits = expLogits.RowSums()[0];

            // Calculate the softmax probabilities
            var softmaxProbabilities = expLogits / sumExpLogits;

            return softmaxProbabilities;
        }
    }

    public class MatrixJsonConverter<T> : JsonConverter where T : struct, IEquatable<T>, IFormattable
    {
        public override bool CanConvert(Type objectType)
        {
            return typeof(Matrix<T>).IsAssignableFrom(objectType);
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
        {
            JArray array = JArray.Load(reader);
            T[,] matrixData = array.ToObject<T[,]>();
            return Matrix<T>.Build.DenseOfArray(matrixData);
        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
        {
            var matrix = (Matrix<T>)value;
            serializer.Serialize(writer, matrix.ToArray());
        }
    }
}
