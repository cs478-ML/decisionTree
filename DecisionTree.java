import java.util.ArrayList;
import java.util.Map.Entry;
import java.util.TreeMap;



public class DecisionTree extends SupervisedLearner {
	
	private Node rootNode;
	
	public DecisionTree(){
		rootNode = null;
	}
	
	private void makeTree(Matrix features, Matrix labels, ArrayList<Double> attributes, Node currNode){ 
		
		String commonNodeLabel = sameLabelNode(features, labels);
		if(commonNodeLabel != null){ //All features have the same label
			currNode.setAttribute(commonNodeLabel);
			currNode.addInstances(labels.rows());
			return;
		}
		if(attributes.size() == 0){//There are no more properties
			currNode.addInstances(labels.rows());
			Node n = maxFeature(features, labels);
			currNode.setAttribute(n.getAttribute());
			return;
		}
		
		//Passed Base cases:
		
		TreeMap<Double, Double> attrEntropy = new TreeMap<Double, Double>();
		
		for(int i = 0; i < features.cols(); i++){
			double tempGain = calcInfoGain(features, labels, i);
			attrEntropy.put((double)i, tempGain);
		}
		
		//print the map
		//		for(Entry<Double, Double> label : attrEntropy.entrySet()) {
		//		      Double key = label.getKey();
		//		      Double value = label.getValue();
		//		
		//		      System.out.println(key + " => " + value);
		//		}
		
		double minGain = Double.MAX_VALUE;
		double splittingAtt = -1;
		
		//Find the minimum entropy
		for(double d : attrEntropy.keySet()){
			double tempGain = attrEntropy.get(d);
			if(tempGain < minGain){
				minGain = tempGain;
				splittingAtt = d;
			}
		}
		
		currNode.setAttribute(features.attrName((int) splittingAtt));
		
		TreeMap<Double, Matrix> labelPrtns = new TreeMap<Double, Matrix>();
		TreeMap<Double, Matrix> featurePrtns = new TreeMap<Double, Matrix>();
		
		for(int i = 0; i < features.rows(); i++){////////changed the index here
			double[] r = features.row(i);
			double attrID = r[(int) splittingAtt];
			if(!labelPrtns.containsKey(attrID)){
				labelPrtns.put(attrID, new Matrix(labels, i, 0, 1, 1));
			}
			else {
				try {
					labelPrtns.get(attrID).add(labels, i, 0, 1);
				} 
				catch (Exception e) {
					System.out.println("Error trying to add to a matrix");
					e.printStackTrace();
				}
			}
			if(!featurePrtns.containsKey(attrID)){
				featurePrtns.put(attrID, new Matrix(features, i, 0, 1, features.cols()));
			}
			else {
				try {
					featurePrtns.get(attrID).add(features, i, 0, 1);
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
			Node newNode = new Node();
			newNode.setAttVal(features.attrValue((int) splittingAtt, d.intValue()));
			currNode.addChild(newNode);
			makeTree(featurePrtns.get(d), labelPrtns.get(d), attributes, newNode);
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
		
		
		//Print out the occurrence map
//		for(Entry<Double, Integer> label : occurrences.entrySet()) {
//            Double key = label.getKey();
//            Integer value = label.getValue();
//
//            System.out.println(key + " => " + value);
//		}
//		
//		System.out.println("m_enum_to_str : " + labels.m_enum_to_str.get(0).get((int)maxLabel));

		return new Node(labels.m_enum_to_str.get(0).get((int)maxLabel));
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
		
		//		double ent = calcEntropy(labels);
		//		System.out.println("Entropy of the system: " + ent);
		
		ArrayList<Double> attributes = new ArrayList<Double>();
		
		for(int i = 0; i < features.cols(); i++){
			attributes.add((double) i);
		}
		
		//this line assumes that if the first column is continuous that all of the data is continuous
		if (features.valueCount(0) == 0)
			discretizeData(features);
		
		rootNode = new Node();
		makeTree(features, labels, attributes, rootNode);
		System.out.println(rootNode.print());
		
		// TODO Auto-generated method stub
//		System.out.println("------------Features------------");
//		features.print();
//		System.out.println("attribute list");
//		for(int i = 0; i < features.m_attr_name.size(); i++){
//			System.out.println(features.m_attr_name.get(i));
//		}
		
//		System.out.println("------------Labels------------");
//		labels.print();
//		System.out.println("attribute list");
//		for(int i = 0; i < labels.m_attr_name.size(); i++){
//			System.out.println(labels.m_attr_name.get(i));
//		}

	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		// TODO Auto-generated method stub

	}
	
	
	private void discretizeData (Matrix features) {
		int numberOfAttributes = features.cols();
		
		for (int c = 0; c < features.cols(); c++) {
			double mean = features.columnMean(c);
						
			for (int r = 0; r < features.rows(); r++) {
				
				//DISCO-TIZE!
				if (features.get(r, c) > mean)
					features.set(r, c, 1);
				else
					features.set(r, c, 0);
			}
		}
	}
}
