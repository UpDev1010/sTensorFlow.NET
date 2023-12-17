using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

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

        public double Train(Matrix<double> xBatch, Matrix<double> yBatchNonOneHot, double learningRate) 
        {
            // forward propagation

            var hiddenLayerInput = xBatch * weightsInputHidden + biasesHidden;
            var hiddenLayerOutput = Sigmoid(hiddenLayerInput);
            var outputLayerInput = hiddenLayerOutput * weightsOutputHidden + biasesOutput;
            var outputLayerOutput = Softmax(outputLayerInput);

            var yBatch = OneHotEncoding(yBatchNonOneHot, outputLayerInput.ColumnCount);

            // backward propagation
            var loss = CrossEntropyLoss(outputLayerOutput, yBatch);

            var outputDelta = outputLayerOutput - yBatch; // a2 - y

            var tt = (outputDelta * weightsOutputHidden.Transpose());
            var hiddenDelta = tt.PointwiseMultiply(hiddenLayerOutput).PointwiseMultiply(1-hiddenLayerOutput);

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
            for(int i = 0; i < yTest.RowCount; i++)
            {
                var actual = yTest.Row(i).MaximumIndex();
                var predicted = predictions.Row(i).MaximumIndex();

                if(actual == predicted)
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
}
