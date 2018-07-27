import java.util.Random;

import weka.attributeSelection.HoldOutSubsetEvaluator;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveByName;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

public class HoldOut {
	static String fileLocation;
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

			seed = 1;
			folds = 10;

			for(int i=0 ; i<seed ; i++) {
				Instances train = dataset.trainCV(folds, i);
				Instances test = dataset.testCV(folds, i);
			}
			Instances [] split = splitTrainTest(dataset, 50);





		}catch (Exception e) {
			System.out.println("Error: " + e);
		}
	}

	public static Instances[] splitTrainTest(Instances data, double p) throws Exception {

		Randomize rand = new Randomize();
		rand.setInputFormat(data);
		rand.setRandomSeed(42);
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
}
