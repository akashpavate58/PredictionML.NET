using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;

namespace ML.NET.SqlTrace
{
    class DataSample
    {
        [LoadColumn(0)]
        public float Ticks { get; set; }

        [LoadColumn(1)]
        public float UsedSpace { get; set; }
    }

    class StoragePrediction
    {
        [ColumnName("Score")]
        public float UsedSpace { get; set; }
    }

    class StorageOverflowPrediction
    {
        [ColumnName("Score")]
        public float Ticks { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {

            MLContext mlContext = new MLContext(seed: 0);
            var model = Train(mlContext);

            int predictFor = 10;

            float predictedSpaceFor50 = Predict(mlContext, model, predictFor);

            Console.WriteLine($"Prediction for Ticks - {predictFor} is {predictedSpaceFor50}");
        }

        private static float Predict(MLContext mlContext, ITransformer model, float ticks)
        {
            var predictionFunction = mlContext.Model.CreatePredictionEngine<DataSample, StoragePrediction>(model);
            var sample = new DataSample
            {
                Ticks = ticks,
                UsedSpace = 0 // To Be Predicted
            };

            var prediction = predictionFunction.Predict(sample);
            return prediction.UsedSpace;
        }

        private static ITransformer Train(MLContext mlContext)
        {
            
            IDataView dataView = mlContext.Data.LoadFromEnumerable(dataSamples);
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "UsedSpace")
                //.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "TicksEncoded", inputColumnName: "Ticks"))

                .Append(mlContext.Transforms.Concatenate("Features", "Ticks"))

                .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression());
            var model = pipeline.Fit(dataView);
            
            return model;
        }

        static List<DataSample> dataSamples = new List<DataSample> {
                new DataSample {
                    Ticks = 1,
                    UsedSpace = 5
                },
                new DataSample {
                    Ticks = 2,
                    UsedSpace = 5
                },
                new DataSample {
                    Ticks = 3,
                    UsedSpace = 6
                },
                new DataSample {
                    Ticks = 4,
                    UsedSpace = 7
                },
                new DataSample {
                    Ticks = 5,
                    UsedSpace = 7
                },
                new DataSample {
                    Ticks = 6,
                    UsedSpace = 8
                },
                new DataSample {
                    Ticks = 7,
                    UsedSpace = 8
                }
            };
    }
}
