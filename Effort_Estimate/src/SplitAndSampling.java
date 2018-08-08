import java.util.Random;
import java.util.Arrays;

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

import org.apache.commons.lang3.ArrayUtils;

public class SplitAndSampling {
	static String fileLocation;
	static Instances dataset;
	static int repeat;
	static int folds;
	static Random rand;
	static DataSource source;
	static double[] resultAvg;
	
	public SplitAndSampling() {
		fileLocation = "./data/miyazaki94.arff";
		
	}
	

	
}
