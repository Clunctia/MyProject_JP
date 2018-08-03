import java.util.Random;

import weka.attributeSelection.HoldOutSubsetEvaluator;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveByName;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.util.Arrays;
import org.apache.commons.lang3.ArrayUtils;
public class HoldOut {
	static String fileLocation = "./data/miyazaki94.arff";
	static Instances dataset;
	static int seed;
	static int folds;
	static Random rand;
	static DataSource source;

	public static void main(String[]args) {
		try {

			source = new DataSource(fileLocation);
			Instances dataset_ID = source.getDataSet();
			RemoveByName rbName = new RemoveByName();
			rbName.setExpression("ID");
			rbName.setInputFormat(dataset_ID);
			dataset = Filter.useFilter(dataset_ID, rbName);
			dataset.setClassIndex(dataset.numAttributes()-1);

			seed = 5;
			folds = 10;
			double percent = 50;

			//			for(int i=0 ; i<seed ; i++) {
			//				Instances train = dataset.trainCV(folds, i);
			//				Instances test = dataset.testCV(folds, i);
			//			}



			//			Print the information about the split data set the first one in the array is train data set and the second one is the test data set.
			//			for(int i = 0 ; i < split.length ; i++ ) {
			//				System.out.println(split[i].toString());
			//				System.out.println(split[i].toSummaryString());
			//			}

			//			classiFier = new Classifier();
			//			classiFier.buildClassifier(split[0]);

//			for(int i = 0 ; i < seed ; i++) {
				Instances [] split = splitTrainTest(dataset, percent);
				
				//save to this 2 arrays
				Instances train = split[0];
				Instances test = split[1];
				
				//get the last column of the Instances array.
				double[] actual_0 = train.attributeToDoubleArray(train.numAttributes()-1);
				double[] actual_1= train.attributeToDoubleArray(test.numAttributes()-1);
				
				double[] actual = ArrayUtils.addAll(actual_0, actual_1);

				LinearRegression lrTrain = new LinearRegression();
				lrTrain.buildClassifier(train);

				Evaluation evalTrain = new Evaluation(train);
				double[] pred_0 = evalTrain.evaluateModel(lrTrain, test);
				
				System.out.println("Use Linear Regression to evaluate the holdout train data");
				System.out.println(evalTrain.toSummaryString());
				System.out.println("------------------------------------------------");

				LinearRegression lrTest = new LinearRegression();
				lrTest.buildClassifier(test);

				Evaluation evalTest = new Evaluation(test);
				double[] pred_1 = evalTest.evaluateModel(lrTest, train);
				
				System.out.println("Use Linear Regression to evaluate the holdout test data");
				System.out.println(evalTest.toSummaryString());
				System.out.println();
				
				double[] predict = ArrayUtils.addAll(pred_0, pred_1);
				
				
				
//			}

		}catch (Exception e) {
			System.out.println("Catch Error: " + e);
		}
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

	public static void holdOut(Classifier clf, Instances data, double p) {



	}
	/*
	 * Extract the last column of train and test data and combine
	 * The combined actual data is used for evaluation of the combined prediction data
	 * How to evaluate? -> Mean Absolute Error average of abs(pred[i] - act[i]) for all data 0 to i 
	 * 
	 */
}
