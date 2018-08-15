import java.util.Random;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.unsupervised.instance.Resample;
public class SplitAndSampling {
	Instances dataset;
	Instances[] result;
	Random rand;
	
	static int repeat;
	
	public Instances[] getResult() {
		return result;
	}
	
	public SplitAndSampling(Instances dataset) throws Exception{
		this.dataset = dataset;
		this.rand = new Random();
		this.result = this.splitAndCombine();
	}
	
	public Instances[] splitAndCombine() throws Exception {
		Instances splited[] = splitIntoThree(dataset);
		
		Randomize randomize = new Randomize();
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
		Instances smallRemain = findRemain(small, smallResample);
		Instances mediumRemain = findRemain(medium, mediumResample);
		Instances largeRemain = findRemain(large, largeResample);
		
		Instances combineResample = combineInstances(smallResample, mediumResample, largeResample);
		
		Instances combineRemain = combineInstances(smallRemain, mediumRemain, largeRemain);
		
		return new Instances[] {combineResample, combineRemain};
		
	}
	
	public static void printInstances(Instances data) {
		for(int i = 0 ; i < data.size() ; i++) {
			System.out.println(data.get(i));
		}
		System.out.println("Size of Data : " + data.size());
	}
	
	public static Instances combineInstances(Instances data1, Instances data2, Instances data3) {
		Instances result = new Instances(data1);
		
		for(int i = 0 ; i < data2.size() ; i++) {
			result.add(data2.get(i));
		}
		
		for(int i = 0 ; i < data3.size() ; i++) {
			result.add(data3.get(i));
		}
		
		return result;
	}
	
	public static Instances findRemain(Instances data, Instances sampledData){
		Instances result = new Instances(data);
		result.delete();
		
		boolean contains = false;
		
		for(int i = 0 ; i < data.size(); i++){
			for(int j = 0 ; j < sampledData.size() ; j++) {
				if(data.get(i).toString().equals(sampledData.get(j).toString())) {
					contains = true;
					break;
				}
			}
			if(!contains) result.add(data.get(i));
			else contains = false;
		}
		
		return result;
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
