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
/*
 * Your project is to answer the following question:
 * Which of 10-fold CV or Hold-out is better to evaluate (and compare) effort estimation models.
 * Fig.3 in the paper shows what functions should be implemented.
 * The white boxes in the figure correspond to functions.
 * "Model Construction" corresponds to building a Linear Regression model.
 * "Performance Measurement" corresponds to calculating a mean absolute error.
 * "Data Preparation for Performance Evaluation" corresponds to the techniques such as 10-fold CV and Hold-out.
 * Please read the paper carefully, and try to implement the remained boxes.
 */