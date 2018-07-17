import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveByName;

public class HoldOut {
	static String fileLocation = "./data/miyazaki94.arff";
	public static void main(String[]args) {
		try {
			
			DataSource source = new DataSource(fileLocation);
			Instances dataset_ID = source.getDataSet();
			RemoveByName rbName = new RemoveByName();
			rbName.setExpression("ID");
			rbName.setInputFormat(dataset_ID);
			Instances dataset = Filter.useFilter(dataset_ID, rbName);
			dataset.setClassIndex(dataset.numAttributes() - 1);
			
			
			
		}catch (Exception e) {
			System.out.println("Error: " + e);
		}
	}
	
}
