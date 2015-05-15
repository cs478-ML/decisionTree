import java.util.ArrayList;
import java.util.Map.Entry;
import java.util.TreeMap;



public class DecisionTree extends SupervisedLearner {
	
	private Node rootNode;
	
	public DecisionTree(){
		rootNode = null;
	}
	
	private void makeTree(Matrix features, Matrix labels, ArrayList<Double> attributes, Node currNode, boolean entropy){ 
		
		String commonNodeLabel = sameLabelNode(features, labels);
		if(commonNodeLabel != null){ //All features have the same label
			currNode.setName(commonNodeLabel);
			currNode.addInstances(labels.rows());
			currNode.setAttrID(labels.m_str_to_enum.get(0).get(commonNodeLabel));
			return;
		}
		if(attributes.size() == 0){//There are no more properties
			currNode.addInstances(labels.rows());
			Node n = maxFeature(features, labels);
			currNode.setName(n.getName());
			currNode.setAttrID(labels.m_str_to_enum.get(0).get(n.getName()));
			return;
		}
		
		//Passed Base cases:
		
		double splittingAtt = -1;
		
		if(entropy){
		
			TreeMap<Double, Double> attrEntropy = new TreeMap<Double, Double>();
			
			for(int i = 0; i < features.cols(); i++){
				double tempGain = calcInfoGain(features, labels, i);
				attrEntropy.put((double)i, tempGain);
			}
			
			double minGain = Double.MAX_VALUE;
			
			//Find the minimum entropy
			for(double d : attrEntropy.keySet()){
				double tempGain = attrEntropy.get(d);
				if(tempGain < minGain){
					minGain = tempGain;
					splittingAtt = d;
				}
			}
		}
		else { //accuracy
			
			TreeMap<Double, Double> attrAccuracy = new TreeMap<Double, Double>();
			
			for(int i = 0; i < attributes.size(); i++){
				double tempGain = calcAccGain(features, labels, attributes.get(i).intValue());
				attrAccuracy.put(attributes.get(i), tempGain);
			}
			
			double maxGain = -1;
			
			//Find the maximum accuracy
			for(double d : attrAccuracy.keySet()){
				double tempGain = attrAccuracy.get(d);
				if(tempGain > maxGain){
					maxGain = tempGain;
					splittingAtt = d;
				}
			}

		}
		
		currNode.setName(features.attrName((int) splittingAtt));
		currNode.setAttrID(splittingAtt);
		
		TreeMap<Double, Matrix> labelPrtns = new TreeMap<Double, Matrix>();
		TreeMap<Double, Matrix> featurePrtns = new TreeMap<Double, Matrix>();
		
		for(int i = 0; i < features.rows(); i++){////////changed the index here
			double[] r = features.row(i);
			double valueID = r[(int) splittingAtt];
			if(!labelPrtns.containsKey(valueID)){
				labelPrtns.put(valueID, new Matrix(labels, i, 0, 1, 1));
			}
			else {
				try {
					labelPrtns.get(valueID).add(labels, i, 0, 1);
				} 
				catch (Exception e) {
					System.out.println("Error trying to add to a matrix");
					e.printStackTrace();
				}
			}
			if(!featurePrtns.containsKey(valueID)){
				featurePrtns.put(valueID, new Matrix(features, i, 0, 1, features.cols()));
			}
			else {
				try {
					featurePrtns.get(valueID).add(features, i, 0, 1);
				}
				catch (Exception e){
					System.out.println("Error trying to add to a matrix");
					e.printStackTrace();
				}
			}
		}
		
		//create the new list of attributes we are interested in
		for(int i = 0; i < attributes.size(); i++){
			if(attributes.get(i) == splittingAtt){
				attributes.remove(i);
			}
		}
		
		//recursive calls on the matrices
		for(Double d : labelPrtns.keySet()){
			
			//Make a copy of the attributes list so it doesn't get messed up from the children
			ArrayList<Double> newAttributeList = new ArrayList<Double>(attributes);
			
			Node newNode = new Node();
			newNode.setParent(features.attrValue((int) splittingAtt, d.intValue()));
			currNode.addChild(newNode);
			makeTree(featurePrtns.get(d), labelPrtns.get(d), newAttributeList, newNode, entropy);
		}
		return;
	}
	
	//Function that checks if all the features have the same label
	// Returns a node with the common label or null if none exists
	private String sameLabelNode(Matrix features, Matrix labels){
		
		double[] currLabelRow = labels.row(0);
		double currNum = currLabelRow[0];
		boolean sameLabels = false;
		
		for(int i = 0; i < labels.rows(); i++){
			sameLabels = false;
			currLabelRow = labels.row(i);
			if(currNum != currLabelRow[0]){
				//System.out.println("Not all the same");
				break;
			}
			sameLabels = true;
		}
		
		if(sameLabels){
			return labels.m_enum_to_str.get(0).get((int)currNum);
		}
		else
			return null;
	}
	
	//Finds the maximum occurring label given the features
	private Node maxFeature(Matrix features, Matrix labels){
		
		TreeMap<Double, Integer> occurrences = new TreeMap<Double, Integer>();
		
		//Calculate the number of occurrences for each label
		for(int i = 0; i < labels.rows(); i++){
			double[] r = labels.row(i);
			if(occurrences.containsKey(r[0])){
				int num = occurrences.get(r[0]);
				occurrences.put(r[0], num + 1);
			}
			else {
				occurrences.put(r[0], 1);
			}
		}
		
		//Find the maximum occurring label
		int max = 0;
		double maxLabel = -1.0;
		for(Double d : occurrences.keySet()){
			int num = occurrences.get(d);
			if(num > max){
				max = num;
				maxLabel = d;
			}
		}

		return new Node(labels.m_enum_to_str.get(0).get((int)maxLabel));
	}

	//Calculates the accuracy of a given data set
	private double calcAccuracy(Matrix labels){
		
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
	
	//Calculates the accuracy gain when splitting on a given attribute
	private double calcAccGain(Matrix features, Matrix labels, int attrIndex){
		
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
	private double calcEntropy(Matrix labels){
		
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
	
	private double calcInfoGain(Matrix features, Matrix labels, int attrIndex){
		
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
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		
		ArrayList<Double> attributes = new ArrayList<Double>();
		
		for(int i = 0; i < features.cols(); i++){
			attributes.add((double) i);
		}
		
		//this line assumes that if the first column is continuous that all of the data is continuous
		if (features.valueCount(0) == 0)
			discretizeData(features);

		
		boolean entropy = true;
		
		rootNode = new Node();
		makeTree(features, labels, attributes, rootNode, entropy);
		System.out.println(rootNode.print());

	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		// TODO Auto-generated method stub
		System.out.println("Features: ");
		for(int i = 0; i < features.length; i++){
			System.out.println(features[i]);
		}
		double prediction = rootNode.classify(features);
		labels[0] = prediction;
		
	}
	
	
	private void discretizeData (Matrix features) {
		
//		for (int c = 0; c < features.cols(); c++) {
//			double min = features.columnMin(c);
//			double max = features.columnMax(c);
//			
//			double interval1 = min + (max - min)/3;
//			double interval2 = interval1 + (max - min)/3;
//						
//			for (int r = 0; r < features.rows(); r++) {
//				
//				//DISCO-TIZE!
//				if (features.get(r, c) < interval1)
//					features.set(r, c, 0);
//				else if (features.get(r, c) < interval2)
//					features.set(r, c, 1);
//				else
//					features.set(r, c, 2);
//			}
//		}
		
		for (int c = 0; c < features.cols(); c++) {
			double mean = features.columnMean(c);
						
			for (int r = 0; r < features.rows(); r++) {
				
				//DISCO-TIZE!
				if (features.get(r, c) > mean)
					features.set(r, c, 1);
				else
					features.set(r, c, 0);
			}
			
			features.m_enum_to_str.get(c).put(0, "0");
			features.m_enum_to_str.get(c).put(1, "1");
			
			features.m_str_to_enum.get(c).put("0", 0);
			features.m_str_to_enum.get(c).put("1", 1);
		}
		return;
		
	}
}


//		double ent = calcEntropy(labels);
//		System.out.println("Entropy of the system: " + ent);

//		double acc = calcAccuracy(labels);
//		System.out.println("Accuracy of the data: " + acc);

/*		
	//print the map for Entropy
for(Entry<Double, Double> label : attrEntropy.entrySet()) {
      Double key = label.getKey();
      Double value = label.getValue();

      System.out.println(key + " => " + value);
}
*/

//System.out.println("------------Features------------");
//features.print();
//System.out.println("attribute list");
//for(int i = 0; i < features.m_attr_name.size(); i++){
//	System.out.println(features.m_attr_name.get(i));
//}

//System.out.println("------------Labels------------");
//labels.print();
//System.out.println("attribute list");
//for(int i = 0; i < labels.m_attr_name.size(); i++){
//	System.out.println(labels.m_attr_name.get(i));
//}

//Print out the occurrence map for Node maxFeature()
//for(Entry<Double, Integer> label : occurrences.entrySet()) {
//    Double key = label.getKey();
//    Integer value = label.getValue();
//
//    System.out.println(key + " => " + value);
//}
//
//System.out.println("m_enum_to_str : " + labels.m_enum_to_str.get(0).get((int)maxLabel));
