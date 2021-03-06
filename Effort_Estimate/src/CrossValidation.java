import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveByName;

public class CrossValidation {
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
			
			
		}catch (Exception e) {
			System.out.println("Error: " + e);
		}
	}
}
