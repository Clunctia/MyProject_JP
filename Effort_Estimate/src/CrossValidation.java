import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LinearRegression;


import java.io.File;
import java.io.IOException;


public class CrossValidation {
	static String fileLocation = "./miyazaki94.arff";

	public static void main(String[]args) {
		try {
			
			//Load Data from the file
			DataSource source = new DataSource(fileLocation);
			Instances dataset = source.getDataSet();
			
			
			dataset.setClassIndex(dataset.numAttributes()-1);
			
			System.out.println("---------------------------------------");

			LinearRegression lr = new LinearRegression();
			lr.buildClassifier(dataset);

			System.out.println(lr);
			
			
		} catch (Exception e) {
		}

	}
}

/* 
 * Note to do:
 * Show mean absolute error with cross validation.
 * Try to use another modeling method.
 * SVM - has several parameters to be tuned.
 * Compare performance between SVM and LinearRegression.
 * 
 * 
 */




