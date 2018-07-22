import java.util.Random;
import weka.classifiers.Classifier;
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
	static Classifier classiFier;
	
		
	public static void main(String[]args) {
		try {
			source = new DataSource(fileLocation);
			Instances dataset_ID = source.getDataSet();
			RemoveByName rbName = new RemoveByName();
			rbName.setExpression("ID");
			rbName.setInputFormat(dataset_ID);
			dataset = Filter.useFilter(dataset_ID, rbName);
			dataset.setClassIndex(dataset.numAttributes()-1);
			
			
			
			
		}catch (Exception e) {
			System.out.println("Error: " + e);
		}
	}
}
