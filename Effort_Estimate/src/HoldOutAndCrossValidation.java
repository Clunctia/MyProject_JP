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
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class HoldOutAndCrossValidation {
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
	// int[] Array1 = new int[] {1,652,5,15,385,4,55,666,13};
	// int[] Array2 = new int[] {2,4658,9,55,-588,10,1083,17};
	// contains = false;
	// List<Integer> results = new ArrayList<Integer>();
	// System.out.println(Array1.length);
	// System.out.println(Array2.length);
	//
	// for(int i=0; i<Array1.length; i++) {
	//     for(int j=0; j<Array2.length; j++) {
	//         if(Array1[i]==Array2[j]) {
	//             contains = true;
	//             break;
	//         }
	//     }
	//     if(!contains) {
	//         results.add(Array1[i]);
	//     }
	//     else{
	//         contains = false;
	//     }
	// }
	//
	// System.out.println(results);
}
