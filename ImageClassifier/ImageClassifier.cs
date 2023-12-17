using MathNet.Numerics.Data.Text;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Tensorflow;

namespace ImageClassifier
{
    public partial class ImageClassifier : Form
    {
        string ImageFilePath;
        public ImageClassifier()
        {
            InitializeComponent();
        }

        private void BtnAnalyzeEmotion_Click(object sender, EventArgs e)
        {
        }

        private void BtnSelectImageFile_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Filter = "Image File (*.jpg, *.png) | *.jpg;*.png";
            if (ofd.ShowDialog() != DialogResult.OK) return;

            ImageFilePath = ofd.FileName;
            pbImage.Load(ImageFilePath);
        }

        private void BtnTrain_Click(object sender, EventArgs e)
        {
            int inputSize = 28 * 28;
            int hiddenSize = 128;
            int outputSize = 10;

            var (xTrain, yTrain) = LoadAndPreprocessData("train.csv");
            var (xTest, yTest) = LoadAndPreprocessData("test.csv");
            var model = new NeuralNetwork(inputSize, hiddenSize, outputSize);

            int epochs = 5;
            double learningRate = 0.001;
            int batchSize = 1;
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int batchStart = 0; batchStart < xTrain.RowCount; batchStart += batchSize)
                {
                    var xBatch = xTrain.SubMatrix(batchStart, batchSize, 0, inputSize);
                    var yBatch = yTrain.SubMatrix(batchStart, batchSize, 0, 1);

                    var loss = model.Train(xBatch, yBatch, learningRate);
                    Console.WriteLine($"Epoch {epoch + 1}, Loss : {loss}");
                }
            }

            double testAccurcy = model.Evaluate(xTest, yTest);
            Console.WriteLine($"Test Accuracy is {testAccurcy * 100}%");

        }

        static (Matrix<double>, Matrix<double>) LoadAndPreprocessData(string filePath)
        {
            var data = DelimitedReader.Read<double>(filePath, false, ",", false, System.Globalization.CultureInfo.InvariantCulture.NumberFormat);
            int inputSize = data.ColumnCount - 1;
            int outputSize = 1;

            var xData = data.SubMatrix(0, data.RowCount, outputSize, inputSize);
            var yData = data.SubMatrix(0, data.RowCount, 0, outputSize);
            return (xData / 255.0, yData / 255.0);
        }
    }
}
