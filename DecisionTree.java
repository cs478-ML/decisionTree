import java.util.ArrayList;
import java.util.Map.Entry;
import java.util.TreeMap;

public class DecisionTree extends SupervisedLearner {
	
	private Node rootNode = new Node();
	private boolean ENTROPY = false;
	
	private void makeTree(Matrix features, Matrix labels, ArrayList<Double> attributes, Node currNode){
		
		String commonNodeLabel = sameLabelNode(features, labels);
				
		//All features have the same label
		if(commonNodeLabel != null){ 
			currNode.setName(commonNodeLabel);
			currNode.addInstances(labels.rows());
			currNode.setAttrID(labels.m_str_to_enum.get(0).get(commonNodeLabel));
			return;
		}
		
		//There are no more properties
		if(attributes.size() == 0){
			currNode.addInstances(labels.rows());
			Node n = maxFeature(features, labels);
			currNode.setName(n.getName());
			currNode.setAttrID(labels.m_str_to_enum.get(0).get(n.getName()));
			return;
		}
		
		double splittingAtt = -1;
		
		if (ENTROPY) {
		
			TreeMap<Double, Double> attrEntropy = new TreeMap<Double, Double>();
			
			for(int i = 0; i < attributes.size(); i++){
				double tempGain = Criterion.calcInfoGain(features, labels, attributes.get(i).intValue());
				attrEntropy.put(attributes.get(i), tempGain);
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
				double tempGain = Criterion.calcAccGain(features, labels, attributes.get(i).intValue());
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
		
		// we decided on a splitting attribute.  Put that info in the node.
		currNode.setName(features.attrName((int) splittingAtt));
		currNode.setAttrID(splittingAtt);
		
		TreeMap<Double, Matrix> labelPrtns = new TreeMap<Double, Matrix>();
		TreeMap<Double, Matrix> featurePrtns = new TreeMap<Double, Matrix>();
		
		for(int i = 0; i < features.rows(); i++){
			
			double[] r = features.row(i);
			double valueID = r[(int) splittingAtt];
			
			try {
				//add to label partition
				if(!labelPrtns.containsKey(valueID))
					labelPrtns.put(valueID, new Matrix(labels, i, 0, 1, 1));
				else
					labelPrtns.get(valueID).add(labels, i, 0, 1);
				
				//add to features partition
				if(!featurePrtns.containsKey(valueID))
					featurePrtns.put(valueID, new Matrix(features, i, 0, 1, features.cols()));
				else
					featurePrtns.get(valueID).add(features, i, 0, 1);					

			} catch (Exception e){
				System.out.println("Error trying to add to a matrix");
				e.printStackTrace();
			}
		}
		
		//Make a copy of the attributes list so it doesn't get messed up from the children
		ArrayList<Double> newAttributeList = new ArrayList<Double>(attributes);
		
		//remove the attribute that we are not interested in
		for(int i = 0; i < newAttributeList.size(); i++){
			if(newAttributeList.get(i) == splittingAtt){
				newAttributeList.remove(i);
			}
		}
		
		//recursive calls on the matrices
		for(Double d : labelPrtns.keySet()){
			
			Node newNode = new Node();
			newNode.setParent(features.attrValue((int) splittingAtt, d.intValue()));
			currNode.addChild(newNode);
			makeTree(featurePrtns.get(d), labelPrtns.get(d), newAttributeList, newNode);
		}
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

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		
		ArrayList<Double> attributes = new ArrayList<Double>();
		
		for(int i = 0; i < features.cols(); i++){
			attributes.add((double) i);
		}
		
		if (features.valueCount(0) == 0) discretizeData(features);
		
		makeTree(features, labels, attributes, rootNode);
		
		// print out our tree
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