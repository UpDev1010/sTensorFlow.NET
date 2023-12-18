using MathNet.Numerics.Data.Text;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Tensorflow;

namespace ImageClassifier
{
    public partial class ImageClassifier : Form
    {
        private string ModelFolderPath = Path.GetDirectoryName(Application.ExecutablePath) +  "/model";
        string ImageFilePath;
        NeuralNetwork NNModel;
        int inputSize = 28 * 28;
        int hiddenSize = 128;
        int outputSize = 10;

        public ImageClassifier()
        {
            InitializeComponent();

            NNModel = new NeuralNetwork(inputSize, hiddenSize, outputSize);
        }
        static Matrix<double> ImageToDoubleArray(string imagePath)
        {
            using (var bitmap = new Bitmap(imagePath))
            {
                int width = bitmap.Width;
                int height = bitmap.Height;

                double[,] pixelValues = new double[1, height * width];

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        System.Drawing.Color pixelColor = bitmap.GetPixel(x, y);

                        pixelValues[0, y * width + x] = pixelColor.B / 255.0;
                    }
                }

                var m = Matrix<double>.Build.DenseOfArray(pixelValues);
                return m;
            }
        }

        static void ResizeImage(string inputPath, string outputPath, int newWidth, int newHeight)
        {
            using (var originalImage = Image.FromFile(inputPath))
            {
                using (var resizedImage = new Bitmap(newWidth, newHeight))
                {
                    using (var graphics = Graphics.FromImage(resizedImage))
                    {
                        graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                        graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
                        graphics.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;
                        graphics.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;

                        graphics.DrawImage(originalImage, 0, 0, newWidth, newHeight);
                    }

                    // Save the resized image
                    resizedImage.Save(outputPath, originalImage.RawFormat);
                }
            }
        }

        public void Log(string str)
        {
            RtbLog.Text += str + System.Environment.NewLine;
        }

        private void BtnAnalyzeEmotion_Click(object sender, EventArgs e)
        {
            string outImagePath = Path.Combine(Path.GetDirectoryName(ImageFilePath), Path.GetFileNameWithoutExtension(ImageFilePath) + "_28_28" + Path.GetExtension(ImageFilePath));
            if (File.Exists(outImagePath))
                File.Delete(outImagePath);

            //ResizeImage(ImageFilePath, outImagePath, 28, 28);

            var data = ImageToDoubleArray(ImageFilePath);
            int result = NNModel.PredictNumber(data);
            Log(result.ToString());
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
            var (xTrain, yTrain) = LoadAndPreprocessData("train.csv");
            var (xTest, yTest) = LoadAndPreprocessData("test.csv");

            int epochs = 5;
            double learningRate = 0.01;
            int batchSize = 1;
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int batchStart = 0; batchStart < xTrain.RowCount; batchStart += batchSize)
                {
                    var xBatch = xTrain.SubMatrix(batchStart, batchSize, 0, inputSize);
                    var yBatch = yTrain.SubMatrix(batchStart, batchSize, 0, 1);

                    var loss = NNModel.Train(xBatch, yBatch, learningRate);
                    Console.WriteLine($"Epoch {epoch + 1}, Loss : {loss}");
                }
            }

            double testAccurcy = NNModel.Evaluate(xTest, yTest);
            Log($"Test Accuracy is {testAccurcy * 100}%");
            Log("model train is finished");

        }

        static (Matrix<double>, Matrix<double>) LoadAndPreprocessData(string filePath)
        {
            var data = DelimitedReader.Read<double>(filePath, false, ",", false, System.Globalization.CultureInfo.InvariantCulture.NumberFormat);
            int inputSize = data.ColumnCount - 1;
            int outputSize = 1;

            var xData = data.SubMatrix(0, data.RowCount, outputSize, inputSize);
            var yData = data.SubMatrix(0, data.RowCount, 0, outputSize);
            return (xData / 255.0, yData);
        }

        private void RtbLog_TextChanged(object sender, EventArgs e)
        {
            RtbLog.SelectionStart = RtbLog.Text.Length;
            RtbLog.ScrollToCaret();
        }

        private void BtnSaveModel_Click(object sender, EventArgs e)
        {
            NNModel.Save(ModelFolderPath);
            Log("model save success.");
        }

        private void BtnLoadModel_Click(object sender, EventArgs e)
        {
            if(NNModel.Load(ModelFolderPath))
            {
                Log("load success.");
                return;
            }

            Log("load failed.");
        }
    }
}
