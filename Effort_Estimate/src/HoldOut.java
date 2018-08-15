import java.util.Random;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveByName;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;
import org.apache.commons.lang3.ArrayUtils;

public class HoldOut {
	static String fileLocation = "./data/miyazaki94.arff";
	static Instances dataset;
	static int repeat;
	static int folds;
	static Random rand;
	static DataSource source;
	static double[] resultAvg;

	public static void main(String[]args) {
		try {

			source = new DataSource(fileLocation);
			Instances dataset_ID = source.getDataSet();
			RemoveByName rbName = new RemoveByName();
			rbName.setExpression("ID");
			rbName.setInputFormat(dataset_ID);
			dataset = Filter.useFilter(dataset_ID, rbName);
			dataset.setClassIndex(dataset.numAttributes()-1);

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

	
}
