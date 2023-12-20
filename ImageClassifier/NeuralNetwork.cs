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
        public Matrix<double> WeightsInputHidden;
        public Matrix<double> BiasesHidden;
        public Matrix<double> WeightsOutputHidden;
        public Matrix<double> BiasesOutput;
        public int BatchSize = 1;
        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, int batchsize)
        {
            WeightsInputHidden = Matrix<double>.Build.Random(inputSize, hiddenSize);
            BiasesHidden = Matrix<double>.Build.Random(batchsize, hiddenSize);
            WeightsOutputHidden = Matrix<double>.Build.Random(hiddenSize, outputSize);
            BiasesOutput = Matrix<double>.Build.Random(batchsize, outputSize);

            BatchSize = batchsize;
        }

        public void Save(string folderPath)
        {
            Directory.CreateDirectory(folderPath);
            string wIHPath = Path.Combine(folderPath, "wIH.bin");
            string bIHPath = Path.Combine(folderPath, "bIH.bin");
            string wOHPath = Path.Combine(folderPath, "wOH.bin");
            string bOHPath = Path.Combine(folderPath, "bOH.bin");

            if (File.Exists(wIHPath)) File.Delete(wIHPath);
            if (File.Exists(bIHPath)) File.Delete(bIHPath);
            if (File.Exists(wOHPath)) File.Delete(wOHPath);
            if (File.Exists(bOHPath)) File.Delete(bOHPath);

            File.WriteAllText(wIHPath, JsonConvert.SerializeObject(WeightsInputHidden.ToArray()));
            File.WriteAllText(bIHPath, JsonConvert.SerializeObject(BiasesHidden.ToArray()));
            File.WriteAllText(wOHPath, JsonConvert.SerializeObject(WeightsOutputHidden.ToArray()));
            File.WriteAllText(bOHPath, JsonConvert.SerializeObject(BiasesOutput.ToArray()));
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

            WeightsInputHidden = JsonConvert.DeserializeObject<Matrix<double>>(File.ReadAllText(wIHPath), settings);
            BiasesHidden = JsonConvert.DeserializeObject<Matrix<double>>(File.ReadAllText(bIHPath), settings);
            WeightsOutputHidden = JsonConvert.DeserializeObject<Matrix<double>>(File.ReadAllText(wOHPath), settings);
            BiasesOutput = JsonConvert.DeserializeObject<Matrix<double>>(File.ReadAllText(bOHPath), settings);

            return true;
        }

        public double Train(Matrix<double> xBatch, Matrix<double> yBatchNonOneHot, double learningRate)
        {
            try
            {
                var hiddenLayerInput = xBatch * WeightsInputHidden + BiasesHidden;
                var hiddenLayerOutput = Sigmoid(hiddenLayerInput);

                var outputLayerInput = hiddenLayerOutput * WeightsOutputHidden + BiasesOutput;
                var outputLayerOutput = Softmax(outputLayerInput);

                var yBatch = OneHotEncoding(yBatchNonOneHot, outputLayerInput.ColumnCount);

                // backward propagation
                var loss = CrossEntropyLoss(outputLayerOutput, yBatch);

                var outputDelta = outputLayerOutput - yBatch; // a2 - y

                var tt = (outputDelta * WeightsOutputHidden.Transpose());
                var hiddenDelta = tt.PointwiseMultiply(hiddenLayerOutput).PointwiseMultiply(1 - hiddenLayerOutput);

                WeightsOutputHidden -= hiddenLayerOutput.Transpose() * outputDelta * learningRate;
                BiasesOutput -= outputDelta * learningRate;
                WeightsInputHidden -= xBatch.Transpose() * hiddenDelta * learningRate;
                BiasesHidden -= hiddenDelta * learningRate;

                return loss;
            }
            catch (Exception ex)
            {
                Console.WriteLine("!!!! Train Error : " + ex.Message);
                return 0;
            }
        }

        public Matrix<double> OneHotEncoding(Matrix<double> x, int newRowSize)
        {
            var retMat = Matrix<double>.Build.Dense(x.RowCount, newRowSize);
            for (int i = 0; i < x.RowCount; i++)
                retMat[i, (int)x[i, 0]] = 1;
            return retMat;
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
            var extendedBiasesHidden = Matrix<double>.Build.DenseOfRowVectors(Enumerable.Repeat(BiasesHidden.Row(0), xTest.RowCount));

            var hiddenLayerInput = (xTest * WeightsInputHidden) + extendedBiasesHidden;
            var hiddenLayerOutput = Sigmoid(hiddenLayerInput);

            var extendedBiasesOutput = Matrix<double>.Build.DenseOfRowVectors(Enumerable.Repeat(BiasesOutput.Row(0), xTest.RowCount));

            var outputLayerInput = hiddenLayerOutput * WeightsOutputHidden + extendedBiasesOutput;
            var outputLayerOutput = Softmax(outputLayerInput);
            return outputLayerOutput;
        }

        public int PredictNumber(Matrix<double> xTest)
        {
            var extendedBiasesHidden = Matrix<double>.Build.DenseOfRowVectors(Enumerable.Repeat(BiasesHidden.Row(0), xTest.RowCount));

            var hiddenLayerInput = (xTest * WeightsInputHidden) + extendedBiasesHidden;
            var hiddenLayerOutput = Sigmoid(hiddenLayerInput);

            var extendedBiasesOutput = Matrix<double>.Build.DenseOfRowVectors(Enumerable.Repeat(BiasesOutput.Row(0), xTest.RowCount));

            var outputLayerInput = hiddenLayerOutput * WeightsOutputHidden + extendedBiasesOutput;
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
                    loss -= actual[i, j] * Math.Log(Math.Min(1, predicted[i, j] + 1e-15)); // Add a small epsilon to prevent log(0)
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

        private Matrix<double> Softmax(Matrix<double> x)
        {
            // Exponentiate the logits
            var expLogits = x.PointwiseExp();

            // Calculate the sum of the exponentiated logits
            var sumExpLogits = expLogits.RowSums();

            var sumExpMat = Matrix<double>.Build.DenseOfColumnVectors(Enumerable.Repeat(sumExpLogits, expLogits.ColumnCount));

            // Calculate the softmax probabilities
            var softmaxProbabilities = expLogits.PointwiseDivide(sumExpMat);
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
