import weka.core.Instances;
import weka.core.SelectedTag;
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

public class StudyOnWeka {
	static String fileLocation = "./data/miyazaki94.arff";

	public static void main(String[] args) {
		try {
			System.out.println("Start the program");
			//Load source file to the program
			DataSource source = new DataSource(fileLocation);
			// Set instance from source data.
			Instances datasetX = source.getDataSet();
			RemoveByName rbname = new RemoveByName();
			rbname.setExpression("ID");
			rbname.setInputFormat(datasetX);
			Instances dataset = Filter.useFilter(datasetX, rbname);
			// Set the index of the data set to the last attribute
			dataset.setClassIndex(dataset.numAttributes() - 1);

			System.out.println("--------------------------Do Linear Regression---------------------------");

			// Call for Linear Regression model to train the model with the data.
			LinearRegression lr = new LinearRegression();
			lr.buildClassifier(dataset);

			System.out.println("The result of Linear Regression");
			System.out.println(lr);

			System.out.println("--------------------------Do Evaluation Model----------------------------");
			
			// Bring the dataset to the evaluation.
			Evaluation eval = new Evaluation(dataset);
			eval.evaluateModel(lr, dataset);
			System.out.println(eval.toSummaryString("Evaluation result: \n", false));

			System.out.println("-----------Do 10 folds cross validation model---------------------------");

			// use dataset after linear regression
			Random rand = new Random(1);
			int folds = 10;
			Evaluation crossEval = new Evaluation(dataset);
			crossEval.crossValidateModel(lr, dataset, folds, rand);
			System.out.println("10 folds Cross Validation result: ");
			System.out.println(crossEval.toSummaryString());

			System.out.println("--------------------------------------");
			System.out.println("Try Support Vector Machine aka SVM");
			//Import LibSVM
			LibSVM svm = new LibSVM();
			//Set the data type of the LibSVM from the SelectedTag function to handle the numeric data.
			svm.setSVMType(new SelectedTag(LibSVM.SVMTYPE_EPSILON_SVR, LibSVM.TAGS_SVMTYPE));
			svm.buildClassifier(dataset);
			
			//Use cross validation to evaluate the result from the LibSVM
			System.out.println("-------------------------------------------------------------------------");
			Evaluation crossEval2 = new Evaluation(dataset);
			crossEval2.crossValidateModel(svm, dataset, folds, rand);
			System.out.println("10 folds cross validation result from LibSVM");
			System.out.println(crossEval2.toSummaryString());
			System.out.println("-------------------------------------------------------------------------");
			//Tuning the Parameter of the LibSVM for better performance.
			
			System.out.println("-------------Use CVParameterSelection---------------");
			CVParameterSelection cvp = new CVParameterSelection();
			cvp.setClassifier(svm);
			cvp.setNumFolds(folds);
			cvp.addCVParameter("C 0.1 0.5 5");
			
			cvp.buildClassifier(dataset);
			
			Evaluation cvEvaluation = new Evaluation(dataset);
			cvEvaluation.crossValidateModel(cvp, dataset, folds, rand);
			System.out.println("10 folds cross validation result of CVParameterSelection");
			System.out.println(cvEvaluation.toSummaryString());
			
			
//			GridSearch grid = new GridSearch();
//			grid.buildClassifier(dataset);
			
			
			System.out.println("------------End of the Program-------------");

		} catch (Exception e) {
			System.out.println("Error occur: " + e);
		}

	}
}

/*
 * NOTE To Do:
 * 1. Tuning the SVM by using the CVParameter class form weka, GridSerarch. study on it.
 * 
 * -Done- 
 * 1. Read miyazaki94.arff data from Java code (you can find a function or a class in Weka for that purpose) 
 * 2. Call LinearRegression of Weka from your Java code to train a model with the data. 
 * 3. Estimate effort values of the data with the trained model. 
 * 4. Calculate absolute errors between the estimate and actual effort values. 
 * At 2 and 3, it is better to use cross-validation if you can (this is not a mandatory task for the present)(6/25/2018)
 * 5.Show mean absolute error with cross validation. Try to use
 * another modeling method. SVM = Support Vector Machine SVM - has several
 * parameters to be tuned. - On Cross validation Compare performance between SVM
 * and LinearRegression.(7/2/2018)
 * 
 */
