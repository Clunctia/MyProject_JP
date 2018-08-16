import java.util.Random;

import org.apache.commons.lang3.ArrayUtils;

import weka.classifiers.BVDecompose;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.meta.CVParameterSelection;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveByName;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class Main {
	static SplitAndSampling splitAndSampling;
	static Instances dataset;
	static String fileLocation = "./data/miyazaki94.arff";
	static String fileLocation2 = "./data/china.arff";
	static int repeat;
	static int folds;
	static int seed;
	static Random rand;
	static DataSource source;
	static double[] resultAvg;
	static double[] holdOutAbs;
	static double[] crossValidationMeanAbsolute;
	static double[] crossValidationEvalMeanAbsolute;
	static double[] evalMeanAbs;
	static double[] libMeanAbs;
	static Instances[] sampleData;

	public static void main (String[]args) throws Exception{
		source = new DataSource(fileLocation2);
		Instances dataset_ID = source.getDataSet();
		Instances dataset_Duration;
		Instances dataset_N_effort;
		RemoveByName rbName = new RemoveByName();
		RemoveByName rbName1 = new RemoveByName();
		RemoveByName rbName2 = new RemoveByName();
		rbName.setExpression("ID");
		rbName.setInputFormat(dataset_ID);
		dataset_Duration = Filter.useFilter(dataset_ID, rbName);
		rbName1.setExpression("Duration");
		rbName1.setInputFormat(dataset_Duration);
		dataset_N_effort = Filter.useFilter(dataset_Duration, rbName1);
		rbName2.setExpression("N_effort");
		rbName2.setInputFormat(dataset_N_effort);
		dataset = Filter.useFilter(dataset_N_effort, rbName2);
		dataset.setClassIndex(dataset.numAttributes()-1);

		int n = 1000;
		evalMeanAbs = new double[n];
		holdOutAbs = new double[n];
		libMeanAbs = new double[n];
		crossValidationMeanAbsolute = new double[n];
		crossValidationEvalMeanAbsolute = new double[n];

		for(int i = 0 ; i < n ; i++) {
			splitAndSampling = new SplitAndSampling(dataset);

			//Get the result of split and sample data form SplitAndSampling class.
			sampleData = splitAndSampling.getResult();
			Instances combineResample = sampleData[0];
			Instances combineRemain = sampleData[1];

			//evalMeanAbs[i] = EvalWithLibSVM(combineResample, combineRemain);
			evalMeanAbs[i] = EvalWithLR(combineResample, combineRemain);

			holdOutAbs[i] = holdOut(combineResample);
			crossValidationMeanAbsolute[i] = crossValidation(combineResample);
		}

		//End 1000 loop
		//Calculate the performance and compare.

		double bias_cv = 0.0;
		for ( int i = 0; i < n; i++) {
			bias_cv += Math.abs(evalMeanAbs[i] - crossValidationMeanAbsolute[i]);
		}
		bias_cv /= 1000.0;

		double bias_ho = 0.0;
		for(int i = 0 ; i < n ; i++) {
			bias_ho += Math.abs(evalMeanAbs[i] - holdOutAbs[i]);
		}
		bias_ho /= 1000.0;

		double var_cv = 0.0;
		var_cv = getVariance(crossValidationMeanAbsolute);

		double var_ho = 0.0;
		var_ho = getVariance(holdOutAbs);

		System.out.println("Bias of CV: " + bias_cv);
		System.out.println("Bias of HO: " + bias_ho);
		System.out.println("Variance of CV: " + var_cv);
		System.out.println("Variance of HO: " + var_ho);
	}

	public static double getVariance(double[] data) {
		int size = data.length;
		double mean = getMean(data);
		double temp = 0;
		for(double a :data)
			temp += (a-mean)*(a-mean);
		return temp/(size-1);
	}

	public static double getMean(double[] data) {
		int size = data.length;
		double sum = 0.0;
		for(double a : data)
			sum += a;
		return sum/size;
	}

	public static double EvalWithLibSVM(Instances training, Instances testing) throws Exception {
		LibSVM svm = new LibSVM();
		svm.setSVMType(new SelectedTag(LibSVM.SVMTYPE_EPSILON_SVR, LibSVM.TAGS_SVMTYPE));
		svm.buildClassifier(training);

		Evaluation eval = new Evaluation(training);
		eval.evaluateModel(svm, testing);
		return eval.meanAbsoluteError();
	}

	public static double EvalWithLR(Instances training, Instances testing) throws Exception {
		LinearRegression lr = new LinearRegression();
		lr.buildClassifier(training);
		Evaluation eval = new Evaluation(training);		
		eval.evaluateModel(lr, testing);
		return eval.meanAbsoluteError();
	}

	public static double crossValidation(Instances dataset) throws Exception {
		folds = 10;
		rand = new Random();

		LinearRegression lr = new LinearRegression();
		lr.buildClassifier(dataset);
		//		System.out.println("Evaluation Cross Validate model");
		Evaluation crossEvaluation = new Evaluation(dataset);
		crossEvaluation.crossValidateModel(lr, dataset, folds, rand);
		//		System.out.println(crossEvaluation.toSummaryString());

		return crossEvaluation.meanAbsoluteError();
	}

	public static double holdOut(Instances dataset) throws Exception{
		repeat = 100;
		resultAvg = new double[repeat];
		//MAEs = Mean Absolute Errors
		double percent = 50, result = 0, MAEs = 0;
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

			//			System.out.println("Use Linear Regression to evaluate the holdout train data");
			//			System.out.println(evalTrain.toSummaryString());
			//			System.out.println("------------------------------------------------");

			lrTest = new LinearRegression();
			lrTest.buildClassifier(test);

			//Predict
			Evaluation evalTest = new Evaluation(test);
			pred_0 = evalTest.evaluateModel(lrTest, train);

			//			System.out.println("Use Linear Regression to evaluate the holdout test data");
			//			System.out.println(evalTest.toSummaryString());


			predict = ArrayUtils.addAll(pred_0, pred_1);

			result = 0;
			for(int j = 0 ; j < predict.length ; j++) {
				result += Math.abs(predict[j]-actual[j]);
			}
			resultAvg[i] = result / actual.length;

		}

		MAEs = 0;
		for(int i = 0 ; i<resultAvg.length ; i++) {
			MAEs += resultAvg[i];
		}
		MAEs = MAEs / resultAvg.length;
		//		System.out.println("The result of all loop: " + sum);

		return MAEs;
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
