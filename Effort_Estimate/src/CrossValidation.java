import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.meta.CVParameterSelection;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveByName;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class CrossValidation {
	static String fileLocation = "./data/miyazaki94.arff";

	public static void main(String[] args) {
		try {
			// Load Data from the file
			System.out.println("Start the program");
			DataSource source = new DataSource(fileLocation);
			// Set instance from source data.
			Instances datasetX = source.getDataSet();
			RemoveByName rbname = new RemoveByName();
			rbname.setExpression("ID");
			rbname.setInputFormat(datasetX);
			Instances dataset = Filter.useFilter(datasetX, rbname);
			// Set the index of the data set to the last attribute
			dataset.setClassIndex(dataset.numAttributes() - 1);

			System.out.println("---------------------------------------");
			System.out.println("------------Do Linear Regression--------------");

			// Call for Linear Regression model to train the model with the data.
			LinearRegression lr = new LinearRegression();
			lr.buildClassifier(dataset);

			System.out.println("The result of Linear Regression");
			System.out.println(lr);

			System.out.println("-----------Do Evaluation Model---------------------------");

			// Bring the dataset to the evaluation.
			Evaluation eval = new Evaluation(dataset);
			eval.evaluateModel(lr, dataset);
			System.out.println(eval.toSummaryString("Evaluation result: \n", false));

			System.out.println("-----------Do 10 folds cross validation model---------------------------");

			// use dataset after linear regression
			Random rand = new Random(1);
			int folds = 10;
			// eval.crossValidateModel(lr, dataset, folds, rand);
			// System.out.println(eval.toSummaryString("10 folds Cross Validation result:
			// \n", false));

			Evaluation crossEval = new Evaluation(dataset);
			System.out.println("-------------------------------------------------------------------------");
			crossEval.crossValidateModel(lr, dataset, folds, rand);
			System.out.println(crossEval.toSummaryString("10 folds Cross Validation result: \n", false));

			System.out.println("--------------------------------------");
			System.out.println("Try Support Vector Machine aka SVM");

			LibSVM svm = new LibSVM();
			// got an error because did't set type of the LibSVM
			svm.setSVMType(new SelectedTag(LibSVM.SVMTYPE_EPSILON_SVR, LibSVM.TAGS_SVMTYPE));
			svm.buildClassifier(dataset);

			Evaluation crossEval2 = new Evaluation(dataset);
			System.out.println("-------------------------------------------------------------------------");
			crossEval2.crossValidateModel(svm, dataset, folds, rand);
			System.out.println(crossEval2.toSummaryString("10 folds Cross Validation with SVM result: \n", false));

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
