import java.util.Random;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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
import weka.filters.unsupervised.instance.Resample;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.math.*;

public class SplitAndSampling {
	static String fileLocation;
	static Instances dataset;
	static int repeat;
	static int folds;
	static Random rand;
	static DataSource source;
	static double[] resultAvg;
	
	public static void main(String[] args) {
		try {
			fileLocation = "./data/miyazaki94.arff";
			source = new DataSource(fileLocation);
			Instances dataset_ID = source.getDataSet();
			RemoveByName rbName = new RemoveByName();
			rbName.setExpression("ID");
			rbName.setInputFormat(dataset_ID);
			dataset = Filter.useFilter(dataset_ID, rbName);
			dataset.setClassIndex(dataset.numAttributes()-1);
			
			System.out.println("Size of all data: " + dataset.size() + "\n");
			
			Instances splited[] = splitIntoThree(dataset);
			
			Randomize randomize = new Randomize();
			rand = new Random();
			randomize.setRandomSeed(rand.nextInt());
			randomize.setInputFormat(dataset);
			
			Instances small = splited[0];
			Instances medium = splited[1];
			Instances large = splited[2];
			
			Instances smallResample, mediumResample, largeResample;
			
			double sizePercentage = 70;
			
			Resample re = new Resample();
			
			re.setInputFormat(small);
			re.setRandomSeed(rand.nextInt());
			re.setNoReplacement(true);
			re.setSampleSizePercent(sizePercentage);
			smallResample = Filter.useFilter(small, re);
			
			re.setInputFormat(medium);
			re.setRandomSeed(rand.nextInt());
			re.setNoReplacement(true);
			re.setSampleSizePercent(sizePercentage);
			mediumResample = Filter.useFilter(medium, re);
			
			re.setInputFormat(large);
			re.setRandomSeed(rand.nextInt());
			re.setNoReplacement(true);
			re.setSampleSizePercent(sizePercentage);
			largeResample = Filter.useFilter(large, re);
			
			//Extract the unseen data that not in the random sampling data set
			
			List<Instance> smallRemain = findRemain(small, smallResample);
			List<Instance> mediumRemain = findRemain(medium, mediumResample);
			List<Instance> largeRemain = findRemain(large, largeResample);
			
			Instances tmp = smallResample;
			
			for(int i = 0 ; i < mediumResample.size() ; i++) {
				tmp.add(mediumResample.get(i));
			}
			
			for(int i = 0 ; i < largeResample.size() ; i++) {
				tmp.add(largeResample.get(i));
			}
			
			System.out.println(tmp.size() + "\n");
			for(int i = 0 ; i < tmp.size() ; i++) {
				System.out.println(tmp.get(i));
			}
			
		}catch (Exception e) {
			System.out.println("Error exception : " + e);
		}
		
	}
	
	public static void printInstances(Instances data) {
		for(int i = 0 ; i < data.size() ; i++) {
			System.out.println(data.get(i));
		}
		System.out.println("Size of Data : " + data.size());
	}
	
	public static List<Instance> findRemain(Instances data, Instances sampledData){
		List<Instance> remain = new ArrayList<Instance>();
		boolean contains = false;
		
		for(int i = 0 ; i < data.size(); i++){
			for(int j = 0 ; j < sampledData.size() ; j++) {
				if(data.get(i).toString().equals(sampledData.get(j).toString())) {
					contains = true;
					break;
				}
			}
			if(!contains) remain.add(data.get(i));
			else contains = false;
		}
		
		return remain;
	}
	
	public static Instances[] split(Instances data, double p) throws Exception {
		Randomize rand = new Randomize();
		Random r = new Random();
		rand.setRandomSeed(r.nextInt());
		rand.setInputFormat(data);
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
	
	public static Instances[] splitIntoThree (Instances data) throws Exception {
		Instances small, medium, large, tmp;
		double sixty = 66;
		double fifty = 50;
		
		//sort the data by the last attribute of the instances (Effort)
		data.sort(data.numAttributes()-1);
		
		RemovePercentage rp = new RemovePercentage();
		rp.setInputFormat(data);
		rp.setPercentage(sixty);
		tmp = Filter.useFilter(data, rp);
		
		rp.setInputFormat(data);
		rp.setPercentage(sixty);
		rp.setInvertSelection(true);
		large = Filter.useFilter(data, rp);		
		
		rp.setInputFormat(tmp);
		rp.setPercentage(fifty);
		small = Filter.useFilter(tmp, rp);
		
		rp.setInputFormat(tmp);
		rp.setPercentage(fifty);
		rp.setInvertSelection(true);
		medium = Filter.useFilter(tmp, rp);
				
		return new Instances[] {small, medium, large};
	}
}
