import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVSaver;
import java.io.File;
import java.io.IOException;


public class Arff2CSV {
	public static void main(String[]args) throws IOException {
		ArffLoader loader = new ArffLoader();
		loader.setSource(new File(""));
		Instances data = loader.getDataSet();
		
		CSVSaver saver = new CSVSaver();
		saver.setInstances(data);
		
		saver.setFile(new File(""));
		saver.writeBatch();
	}
}
