import java.util.Random;

import org.apache.commons.lang3.ArrayUtils;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveByName;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class Main {
	static SplitAndSampling splitAndSampling;
	static Instances dataset;
	static String fileLocation = "./data/miyazaki94.arff";
	static int repeat;
	static int folds;
	static int seed;
	static Random rand;
	static DataSource source;
	static double[] resultAvg;
	static double[] sumHoldOutResult;
	static double[] holdOutResult;
	static Instances[] sampleData;
	

	public static void main (String[]args) throws Exception{
		
		source = new DataSource(fileLocation);
		Instances dataset_ID = source.getDataSet();
		RemoveByName rbName = new RemoveByName();
		rbName.setExpression("ID");
		rbName.setInputFormat(dataset_ID);
		dataset = Filter.useFilter(dataset_ID, rbName);
		dataset.setClassIndex(dataset.numAttributes()-1);
		
		splitAndSampling = new SplitAndSampling(dataset);
		
		//Get the result of split and sample data form SplitAndSampling class.
		sampleData = splitAndSampling.getResult();
		Instances combineResample = sampleData[0];
		Instances combineRemain = sampleData[1];
		
		LinearRegression lrResample = new LinearRegression();
		lrResample.buildClassifier(combineResample);
		
		Evaluation eval = new Evaluation(combineResample);
		eval.evaluateModel(lrResample, combineRemain);
		
		crossValidation(combineResample);
		holdOut(combineResample);
		
	}
	
	public static void crossValidation(Instances dataset) throws Exception {
		seed = 1;
		folds = 10;
		rand = new Random();
		
		LinearRegression lr = new LinearRegression();
		lr.buildClassifier(dataset);
		System.out.println("Evaluation Cross Validate model");
		Evaluation crossEvaluation = new Evaluation(dataset);
		crossEvaluation.crossValidateModel(lr, dataset, folds, rand);
		System.out.println(crossEvaluation.toSummaryString());
		
		
		System.out.println("Evaluate Linear Regression");
		Evaluation evaluation = new Evaluation(dataset);
		evaluation.evaluateModel(lr, dataset);
		System.out.println(evaluation.toSummaryString());
		
	}
	
	public static void holdOut(Instances dataset) throws Exception{
		repeat = 100;
		folds = 10;
		resultAvg = new double[repeat];
		double percent = 50, result = 0, sum = 0;
		double[] actual_0;
		double[] actual_1;
		double[] actual;
		double[] pred_1;
		double[] pred_0;
		double[] predict;
		Instances[] split;
		Instances train, test;
		LinearRegression lrTrain, lrTest;

		for(int i = 0 ; i < repeat ; i++) {
			split = splitTrainTest(dataset, percent);

			//save to this 2 arrays
			train = split[0];
			test = split[1];

			//get the last column of the Instances array is it an actual value?.
			actual_0 = train.attributeToDoubleArray(train.numAttributes()-1);
			actual_1= test.attributeToDoubleArray(test.numAttributes()-1);

			actual = ArrayUtils.addAll(actual_0, actual_1);

			lrTrain = new LinearRegression();
			lrTrain.buildClassifier(train);

			//Predict
			Evaluation evalTrain = new Evaluation(train);

			pred_1 = evalTrain.evaluateModel(lrTrain, test);

			System.out.println("Use Linear Regression to evaluate the holdout train data");
			System.out.println(evalTrain.toSummaryString());
			System.out.println("------------------------------------------------");

			lrTest = new LinearRegression();
			lrTest.buildClassifier(test);

			//Predict?
			Evaluation evalTest = new Evaluation(test);
			pred_0 = evalTest.evaluateModel(lrTest, train);

			System.out.println("Use Linear Regression to evaluate the holdout test data");
			System.out.println(evalTest.toSummaryString());
			

			predict = ArrayUtils.addAll(pred_0, pred_1);
			
			result = 0;
			for(int j = 0 ; j < predict.length ; j++) {
				result += Math.abs(predict[j]-actual[j]);
			}
			resultAvg[i] = result / actual.length;

		}
		
		sum = 0;
		for(int i = 0 ; i<resultAvg.length ; i++) {
			sum += resultAvg[i];
		}
		sum = sum / resultAvg.length;
		
		System.out.println("The result of all loop: " + sum);
	}
	
	public static Instances[] splitTrainTest(Instances data, double p) throws Exception {

		Randomize rand = new Randomize();
		Random r = new Random();
		rand.setRandomSeed(r.nextInt());
		rand.setInputFormat(data);
		data = Filter.useFilter(data, rand);

		RemovePercentage rp = new RemovePercentage();
		rp.setInputFormat(data);
		rp.setPercentage(p);
		Instances train = Filter.useFilter(data, rp);

		rp = new RemovePercentage();
		rp.setInputFormat(data);
		rp.setPercentage(p);
		rp.setInvertSelection(true);
		Instances test = Filter.useFilter(data, rp);

		return new Instances[] {train, test};
	}
}
