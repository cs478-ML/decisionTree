import java.util.TreeMap;


public class Criterion {
	
	//Calculates the accuracy of a given data set
	static double calcAccuracy(Matrix labels){
			
		double accuracy = 0;
		
		double totalInstances = labels.rows();
		
		TreeMap<Double, Integer> labelInstances = new TreeMap<Double, Integer>();
		
		for(int i = 0; i < labels.rows(); i++){
			double[] r = labels.row(i);
			if(labelInstances.containsKey(r[0])){
				int num = labelInstances.get(r[0]);
				labelInstances.put(r[0], num + 1);
			}
			else {
				labelInstances.put(r[0], 1);
			}
		}
		
		double majority = -1;
		
		for(Double d : labelInstances.keySet()){
			if(majority < labelInstances.get(d)){
				majority = labelInstances.get(d);
			}
		}
		
		accuracy = majority / totalInstances;
		
		return accuracy;
	}
		

	static double calcAccGainExtended(Matrix features, Matrix labels, int att1, int att2) {
		double accGain = 0;
		
//		attr.toString() + ',' + Double.toString(splittingAtt)
		
		TreeMap<String, Matrix> partition = new TreeMap<String, Matrix>();
		
		for (int i = 0; i < features.rows(); i++) {
			double[] r = features.row(i);
			String key = Double.toString(r[att1]) + ',' + Double.toString(r[att2]);
			if (!partition.containsKey(key)) {
				partition.put(key, new Matrix(labels, i, 0, 1, 1));
			} else {
				try {
					partition.get(key).add(labels, i, 0, 1);
				} catch (Exception e) {
					System.out.println("error adding to matrix");
					e.printStackTrace();
				}
			}
		}
		
		double totalAttributes = labels.rows();
		
		for(String key : partition.keySet()){
			double numerator = partition.get(key).rows();
			double proportion = numerator / totalAttributes;
			accGain += proportion * calcAccuracy(partition.get(key));
		}
		
		return accGain;	
	}
	
	//Calculates the accuracy gain when splitting on a given attribute
	static double calcAccGain(Matrix features, Matrix labels, int attrIndex){
		
		double accGain = 0;
		
		TreeMap<Double, Matrix> partition = new TreeMap<Double, Matrix>();
		
		for(int i = 0; i < features.rows(); i++){
			double[] r = features.row(i);
			if(!partition.containsKey(r[attrIndex])){
				partition.put(r[attrIndex], new Matrix(labels, i, 0, 1, 1));
			}
			else {
				try {
					partition.get(r[attrIndex]).add(labels, i, 0, 1);
				} 
				catch (Exception e) {
					System.out.println("Error trying to add to a matrix");
					e.printStackTrace();
				}
			}
		}
		
		double totalAttributes = labels.rows();
		
		for(Double d : partition.keySet()){
			double numerator = partition.get(d).rows();
			double proportion = numerator / totalAttributes;
			accGain += proportion * calcAccuracy(partition.get(d));
		}
		
		return accGain;
	}
	//Calculates the entropy for a given attribute for each class
	static double calcEntropy(Matrix labels){
		
		double entropy = 0;
		
		double totalInstances = labels.rows();
		TreeMap<Double, Integer> labelInstances = new TreeMap<Double, Integer>();
		
		for(int i = 0; i < labels.rows(); i++){
			double[] r = labels.row(i);
			if(labelInstances.containsKey(r[0])){
				int num = labelInstances.get(r[0]);
				labelInstances.put(r[0], num + 1);
			}
			else {
				labelInstances.put(r[0], 1);
			}
		}
		
		for(Double d : labelInstances.keySet()){
			double proportion = labelInstances.get(d)/totalInstances;
			entropy = entropy - proportion * (Math.log(proportion) / Math.log(2));
		}
		
		return entropy;
	}
	
	static double calcInfoGain(Matrix features, Matrix labels, int attrIndex){
		
		double infoGain = 0;
		
		TreeMap<Double, Matrix> partition = new TreeMap<Double, Matrix>();
		
		for(int i = 0; i < features.rows(); i++){////////changed the index here
			double[] r = features.row(i);
			if(!partition.containsKey(r[attrIndex])){
				partition.put(r[attrIndex], new Matrix(labels, i, 0, 1, 1));
			}
			else {
				try {
					partition.get(r[attrIndex]).add(labels, i, 0, 1);
				} 
				catch (Exception e) {
					System.out.println("Error trying to add to a matrix");
					e.printStackTrace();
				}
			}
		}
		
		double totalAttributes = labels.rows();
		
		for(Double d : partition.keySet()){
			double numerator = partition.get(d).rows();
			double proportion = numerator / totalAttributes;
			infoGain += proportion * calcEntropy(partition.get(d));
		}
		return infoGain;
	}
	
}
