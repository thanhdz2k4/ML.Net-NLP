using System;
using System.IO;
using CsvHelper;
using System.Globalization;
using System.Collections.Generic;
using System.Dynamic;
using static System.Runtime.InteropServices.JavaScript.JSType;
using System.Collections;
using System.Diagnostics;
using Microsoft.ML;
using Tokenization;
using System.Text;
using System.Text.RegularExpressions;
using ConsoleApp1;
using TextProcedure;
using CsvHelper.Configuration;

using Microsoft.ML.Data;

class Program
{
    static void Main(string[] args)
    {
        // Define the path to your CSV file
        string filePath = @"E:\download and destop\New folder (2)\archive\reviews_data.csv";

         // after fillter with rating and revire
         string savePath = @"E:\download and destop\New folder (2)\archive\filtered_reviews_data.csv";

         // after extract 
         string savePath2 = @"E:\download and destop\New folder (2)\archive\filtered4_special_character_reviews_data.csv";

         // after lower 
         string savePath3 = @"E:\download and destop\New folder (2)\archive\filtered8_special_character_reviews_data.csv";

        // after assign label 
        //string savePath4 = @"E:\download and destop\New folder (2)\archive\filtered9_special_character_reviews_data.csv";

        // ----------------------------------------- new dataset -----------------------------------------------------------------------------------------

        // All fields
        string savePatha = @"E:\download and destop\New folder (2)\archive (2)\Restaurant reviews.csv";

        // review, rating and sentiment
        string savePatha1 = @"E:\download and destop\New folder (2)\archive (2)\dataset.csv";

        // after removing null rating
        string savePatha2 = @"E:\download and destop\New folder (2)\archive (2)\dataset4.csv";

        // after asigning abel
        string savePatha3 = @"E:\download and destop\New folder (2)\archive (2)\dataset5.csv";

        // atter lower text
        string savePatha4 = @"E:\download and destop\New folder (2)\archive (2)\dataset6.csv";


        List<Review> data = ReadCSV(savePatha3);


        //PrintData(data);





        //TrainingData1(data);
      

        Test();

        Console.WriteLine("Done");




    }

    static string RemovePunctuation(string input)
    {
        StringBuilder sb = new StringBuilder();
        foreach (char c in input)
        {
            if (!char.IsPunctuation(c))
            {
                sb.Append(c);
            }
        }
        return sb.ToString();
    }

    private static void Test()
    {
        var feedbackSamples = new[]
{
    "The pumpkin spice latte was delightful! I can't get enough of it.",
    "I visited Starbucks, and the service was incredibly slow. I was very disappointed.",
    "The coffee was great, but the atmosphere was a bit too loud for my liking.",
    "I love the seasonal flavors! They always bring something new and exciting.",
    "My experience was average; the coffee was fine, but nothing extraordinary.",
    "The staff was friendly, but my order was wrong. I had to wait again for it to be corrected.",
    "I enjoy sitting in the cozy corner with a book and a hot drink. A perfect afternoon!",
    "The new holiday drinks are fantastic! I can’t wait to try them all.",
    "Unfortunately, my last visit was a letdown. The mocha was too sweet and left a bad aftertaste.",
    "I had a wonderful time chatting with friends over coffee. The vibe is always welcoming.",
    "The location is great, but parking is a hassle. It often discourages me from visiting.",
    "The iced coffee was refreshing, especially on a hot day. I'll be back for more!",
    "I don't usually drink coffee, but the teas here are superb! I highly recommend them.",
    "I waited too long in line, and when I got my drink, it was lukewarm.",
    "Starbucks is my go-to place for meetings. It's convenient and the Wi-Fi is reliable.",
    "I've always had good experiences here. The baristas know their stuff!",
    "I tried the new oat milk option, and it was delicious! A great alternative.",
    "I was disappointed with the cleanliness of the shop. It could use some attention.",
    "The loyalty program is fantastic! I love getting rewards for my purchases.",
    "Every time I go, I find something new to love. Highly recommend!",
    "i ordered a grande tea and they used only one tea bag, the same as a tall tea. what is the extra charge for in the grande size? water? come on, give me a break.",
    "i order the same things at many starbucks in california. this is the only starbucks that charges me more for the same product. i always order a grande americano with steamed breve. i am charged 65 cents more. i am told it is everything from the breve to the labor. everywhere else it is $2.55."
};
        string modelPath = @"E:\download and destop\New folder (2)\archive\SentimentModel5.zip";

        for (int i = 0; i < feedbackSamples.Length; i++)
        {
            Console.WriteLine(feedbackSamples[i] + " :   " + PredictFromModel(modelPath, feedbackSamples[i]));

        }
    }

    public static void TrainingData(List<Review> reviews)
    {
        var mlContext = new MLContext();

        // Load data from List directly
        var data = mlContext.Data.LoadFromEnumerable(reviews);

        // Split data into train and test sets (80% train, 20% test)
        var splitData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
        var trainData = splitData.TrainSet;
        var testData = splitData.TestSet;

        // Data processing pipeline
        var pipeline = mlContext.Transforms.Text.NormalizeText(outputColumnName:"TextNormalize",inputColumnName: nameof(Review.ReviewText),
                keepDiacritics: false, keepPunctuations: false, keepNumbers: false)
           .Append(mlContext.Transforms.Text.TokenizeIntoWords(outputColumnName: "TokenizeIntoWords", "TextNormalize"))
            .Append(mlContext.Transforms.Text.RemoveDefaultStopWords(outputColumnName: "TokensRemoveStopWords", "TokenizeIntoWords"))
            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", "TextNormalize"))
            .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", nameof(Review.Sentiment)))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        // Train model on training data
        var model = pipeline.Fit(trainData);

        // Evaluate the model using test data
        var predictions = model.Transform(testData);
        var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score");

        // Output evaluation metrics
        Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy}");
        Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy}");
        Console.WriteLine($"LogLoss: {metrics.LogLoss}");

        // Save the model
        string modelPath = @"E:\download and destop\New folder (2)\archive\SentimentModel5.zip";
        mlContext.Model.Save(model, trainData.Schema, modelPath);

        Console.WriteLine($"Model saved to: {modelPath}");


    }

    public static void TrainingData1(List<Review> reviews)
    {
        var mlContext = new MLContext();

        // Load data from List directly
        var data = mlContext.Data.LoadFromEnumerable(reviews);

        // Split data into train and test sets (80% train, 20% test)
        var splitData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
        var trainData = splitData.TrainSet;
        var testData = splitData.TestSet;

        // Data processing pipeline
        var pipeline = mlContext.Transforms.Text.NormalizeText(outputColumnName: "TextNormalize", inputColumnName: nameof(Review.ReviewText),
                keepDiacritics: false, keepPunctuations: false, keepNumbers: true)
            .Append(mlContext.Transforms.Text.TokenizeIntoWords(outputColumnName: "TokenizeIntoWords", "TextNormalize"))
            .Append(mlContext.Transforms.Text.RemoveDefaultStopWords(outputColumnName: "TokensRemoveStopWords", "TokenizeIntoWords"))
            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", "TokensRemoveStopWords")
            .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", nameof(Review.Sentiment)))
            .Append(mlContext.Transforms.NormalizeMinMax("Features")) // Normalizing features can help certain models
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features")) // Using LbfgsLogisticRegression
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel")));

        // Train model on training data
        var model = pipeline.Fit(trainData);

        // Evaluate the model using test data
        var predictions = model.Transform(testData);
        var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score");

        // Output evaluation metrics
        Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy}");
        Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy}");
        Console.WriteLine($"LogLoss: {metrics.LogLoss}");

        // Save the model
        string modelPath = @"E:\download and destop\New folder (2)\archive\SentimentModel5.zip";
        mlContext.Model.Save(model, trainData.Schema, modelPath);

        Console.WriteLine($"Model saved to: {modelPath}");
    }


    private static string PredictFromModel(string modelPath, string review)
    {
        var mlContext = new MLContext();
        DataViewSchema modelSchema;
        var loadedModel = mlContext.Model.Load(modelPath, out modelSchema);

        // Prediction example with loaded model
        var predictionEngine = mlContext.Model.CreatePredictionEngine<Review, ReviewPrediction>(loadedModel);
        var sampleReview = new Review { ReviewText = review };
        var prediction = predictionEngine.Predict(sampleReview);

        return prediction.PredictedSentiment;
    }


    // Define the prediction class
    public class ReviewPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedSentiment { get; set; }
        public float[] Score { get; set; }
    }

    private static List<Review> ToLower(List<Review> data)
        {
            List<Review> result = new List<Review>();
            foreach (Review rv in data)
            {
                Review rv2 = new Review(rv.ReviewText.ToLower(), rv.Rating);
                result.Add(rv2);
            }
            return result;
        }

    public static void SaveReviewsToCsv(List<Review> reviews, string filePath)
        {
            using (var writer = new StreamWriter(filePath))
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                csv.WriteRecords(reviews); // Ghi danh sách vào tệp CSV
            }
        }

    public static List<Review> ExtractCharacterCSV(List<Review> data)
        {
            List<Review> data2 = new List<Review>();

            foreach (Review rv in data)
            {

                string x = rv.ReviewText;
                x = x.Replace("\u2019", "'");
                string y = TextExpander.ExpandContractions(x);

                Review review = new Review(y, rv.Rating);
                data2.Add(review);

            }

            return data2;
        }

  
    private static void PrintData(List<ExpandoObject> data)
        {
            foreach (dynamic obj in data)
            {
                Console.WriteLine(obj.Review);
                Console.WriteLine(obj.Rating);
                Console.WriteLine("-------------------------");
            }
        }

    private static void PrintData(List<Review> data)
        {
            foreach (Review obj in data)
            {
            
                Console.WriteLine(obj.ReviewText);
                Console.WriteLine(obj.Rating);
                Console.WriteLine(obj.Sentiment);
                Console.WriteLine("-------------------------");
            }
        }

    private static void RemoveStopWords(string textParamenter)
        {
            var context = new MLContext();

            var emptyData = new List<TextData>();

            var data = context.Data.LoadFromEnumerable(emptyData);

            var tokenization = context.Transforms.Text.TokenizeIntoWords("Tokens", "Text", separators: new[] { ' ', ',', '.', '*' })
                .Append(context.Transforms.Text.RemoveDefaultStopWords("Tokens", "Tokens",
                Microsoft.ML.Transforms.Text.StopWordsRemovingEstimator.Language.English));

            var model = tokenization.Fit(data);

            var engine = context.Model.CreatePredictionEngine<TextData, TextTokens>(model);

            var text = engine.Predict(new TextData { Text = textParamenter });

            PrintTokens("Removing stop words", text);
        }

    private static void PrintTokens(string tokenType, TextTokens tokens)
        {
            Console.WriteLine($"\n{tokenType}:");

            var sb = new StringBuilder();
            foreach (var token in tokens.Tokens)
            {
                sb.AppendLine(token.ToString());
            }

            Console.WriteLine(sb.ToString());
        }

    private static void CheckDuplicateData(List<ExpandoObject> data)
        {
            var duplicateRecords = data
                    .GroupBy(record => new
                    {
                        Review = ((IDictionary<string, object>)record)["Review"],
                        Rating = ((IDictionary<string, object>)record)["Rating"]
                    })
                    .Where(group => group.Count() > 1)
                    .ToList();

            // Output duplicates
            if (duplicateRecords.Count > 0)
            {
                Console.WriteLine("Duplicate records found:");
                foreach (var group in duplicateRecords)
                {
                    Console.WriteLine($"Review: {group.Key.Review}, Rating: {group.Key.Rating}, Count: {group.Count()}");
                }
            }
            else
            {
                Console.WriteLine("No duplicate records found.");
            }
        }

    public static List<Review> ReadCSV(string filePath)
        {
            using (var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            using (var reader = new StreamReader(fileStream))
            using (var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                HasHeaderRecord = true,
                MissingFieldFound = null, // Ignore missing fields if any
                HeaderValidated = null, // Ignore header validation
                BadDataFound = null // Ignore bad data
            }))
            {
                csv.Context.RegisterClassMap<ReviewMap>();

                // Load records, handle empty Sentiment
                var records = csv.GetRecords<Review>().Select(r => new Review
                {
                    ReviewText = r.ReviewText,
                    Rating = r.Rating,
                    Sentiment = string.IsNullOrEmpty(r.Sentiment) ? "Unknown" : r.Sentiment // Default to "Unknown" if empty
                }).ToList();

                return records;
            }
        }

    
    private static void CountNULLRatingAndReview(List<ExpandoObject> records)
        {
            int countNULLReview = 0;
            int countNULLRating = 0;


            // Process each dynamic record
            foreach (dynamic record in records)
            {
                //Console.WriteLine(record.Rating);
                if (record.Rating == "N/A")
                {
                    countNULLRating++;

                }

                if (record.Review == "No Review Text")
                {
                    countNULLReview++;

                }

                if (record.Review.Length < 60)
                {
                    Console.WriteLine(record.Review);
                }
            }
            Console.WriteLine("Rating's null: " + countNULLRating);
            Console.WriteLine("Review's null:" + countNULLReview);
        }


    private static void CountNULLRating(List<Review> records)
    { 
        List<Review> reviews = new List<Review>();
        int countNULLRating = 0;
        foreach (Review record in records)
        {
            if (record.Rating == null)
            { 
                Console.WriteLine(record.ReviewText); 
                Console.WriteLine(record.Rating);
                
                countNULLRating++;
            } else
            {
                reviews.Add(record);
            }
          
            
        }
        Console.WriteLine(reviews.Count);
        SaveReviewsToCsv(reviews, @"E:\download and destop\New folder (2)\archive (2)\dataset4.csv");
    }
    private static List<Review> AssignLabel(List<Review> data)
        {
            List<Review> ret = new List<Review>();
            foreach (Review review in data)
            {
                Review rv = new Review(review.ReviewText, review.Rating, LabelData.ConvertRatingToSentiment((float)review.Rating));
                ret.Add(rv);
            }
            return ret;

        }


    
}


