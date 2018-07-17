import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.meta.GridSearch;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveByName;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class HoldOutAndCrossValidation {
	static String fileLocation = "./data/miyazaki94.arff";
	public static void main(String[]args) {
		
	}

}
