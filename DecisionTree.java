import java.util.ArrayList;
import java.util.TreeMap;



public class DecisionTree extends SupervisedLearner {
	
	private Node rootNode = null;
	
	boolean EXTENDED = false;
	boolean ENTROPY = false;
	
	private void makeTree(Matrix features, Matrix labels, ArrayList<Double> attributes, Node currNode){ 
		
		//Base Cases for Recursion
		
		String labelForAllNodes = sameLabelNode(features, labels);
		//Do all of the instances have the same class/label?
		if(labelForAllNodes != null){
			currNode.setName(labelForAllNodes);
			currNode.addInstances(labels.rows());
			currNode.setAttrID(labels.m_str_to_enum.get(0).get(labelForAllNodes));
			return;
		}
		
		//Are there any instances left?
		if(attributes.size() == 0){
			String maxLabel = maxAttrVal(labels);
			currNode.setName(maxLabel);
			currNode.addInstances(labels.rows());
			currNode.setAttrID(labels.m_str_to_enum.get(0).get(maxLabel));
			return;
		}
		
		//Passed Base cases:
		
		double splittingAtt = -1;
		double splittingAtt2 = -1;
		boolean extend = false;
		
		if(ENTROPY){
		
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
			
			double wholeSetAccuracy = Criterion.calcAccuracy(labels);
						
			double maxGain = -1;

			for(double attr : attributes){
				double tempGain = Criterion.calcAccGain(features, labels, (int) attr);
				if (tempGain > maxGain) {
					maxGain = tempGain;
					splittingAtt = attr;
				}
			}
			
			if(maxGain <= wholeSetAccuracy){
				
				if (EXTENDED) {
					for (double attr: attributes) {
						if (attr == splittingAtt) continue;
						double tempGain = Criterion.calcAccGainExtended(features, labels, (int) splittingAtt, (int) attr);
						if (tempGain > maxGain) {
							maxGain = tempGain;
							splittingAtt2 = attr;
						}
					}
					
					if(maxGain <= wholeSetAccuracy){
						String maxLabel = maxAttrVal(labels);
						currNode.setName(maxLabel);
						currNode.setAttrID(labels.m_str_to_enum.get(0).get(maxLabel));
						return;
					}
					
					//create the new list of attributes we are interested in
					for(int i = 0; i < attributes.size(); i++){
						if(attributes.get(i) == splittingAtt2){
							attributes.remove(i);
						}
					}
				} else {
					String maxLabel = maxAttrVal(labels);
					currNode.setName(maxLabel);
					currNode.setAttrID(labels.m_str_to_enum.get(0).get(maxLabel));
					return;
				}
			}

		}
		
		TreeMap<Double, Matrix> labelPrtns = new TreeMap<Double, Matrix>();
		TreeMap<Double, Matrix> featurePrtns = new TreeMap<Double, Matrix>();
		
		
		for(int i = 0; i < features.rows(); i++){////////changed the index here
			double[] r = features.row(i);
			double valueID = r[(int) splittingAtt];
			
			try {
				if(!labelPrtns.containsKey(valueID)){
					labelPrtns.put(valueID, new Matrix(labels, i, 0, 1, 1));
				}
				else {
					labelPrtns.get(valueID).add(labels, i, 0, 1);
				}
				if(!featurePrtns.containsKey(valueID)){
					featurePrtns.put(valueID, new Matrix(features, i, 0, 1, features.cols()));
				}
				else {
					featurePrtns.get(valueID).add(features, i, 0, 1);
				}
			} 
			catch (Exception e) {
				System.out.println("Error trying to add to a matrix");
				e.printStackTrace();
			}
		}
		
		//Check to make sure all the values were accounted for:
		for(int i = 0; i < features.valueCount((int)splittingAtt); i++){
			double missingVal = i;
			if(!featurePrtns.containsKey(missingVal)){
				featurePrtns.put(missingVal, null);
				labelPrtns.put(missingVal, null);
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
			
			if (!extend) {
				//Make a copy of the attributes list so it doesn't get messed up from the children
				ArrayList<Double> newAttributeList = new ArrayList<Double>(attributes);
				
				currNode.setName(features.attrName((int) splittingAtt));
				currNode.setAttrID(splittingAtt);
				
				Node newNode = new Node();
				newNode.setParent(features.attrValue((int) splittingAtt, d.intValue()));
				currNode.addChild(newNode);
				if(featurePrtns.get(d) != null){
					makeTree(featurePrtns.get(d), labelPrtns.get(d), newAttributeList, newNode);
				}
				else {
					String maxLabel = maxAttrVal(labels);
					newNode.setName(maxLabel);
					newNode.addInstances(0);
					newNode.setAttrID(labels.m_str_to_enum.get(0).get(maxLabel));
					//return; ?
				}
				
			} else {
				
//				currNode.setName(/*concatenate the two strings*/);
//				currNode.setAttrID(splittingAtt, #2);s
				//partition the partitions based on the 2nd attr
				TreeMap<Double, Matrix> labelPrtns2 = new TreeMap<Double, Matrix>();
				TreeMap<Double, Matrix> featurePrtns2 = new TreeMap<Double, Matrix>();
				
				
				for(int i = 0; i < featurePrtns.get(d).rows(); i++){////////changed the index here
					double[] r = featurePrtns.get(d).row(i);
					double valueID = r[(int) splittingAtt2];
					
					try {
						if(!labelPrtns2.containsKey(valueID)){
							labelPrtns2.put(valueID, new Matrix(labelPrtns.get(d), i, 0, 1, 1));
						}
						else {
							labelPrtns2.get(valueID).add(labelPrtns.get(d), i, 0, 1);
						}
						if(!featurePrtns2.containsKey(valueID)){
							featurePrtns2.put(valueID, new Matrix(featurePrtns.get(d), i, 0, 1, featurePrtns.get(d).cols()));
						}
						else {
							featurePrtns2.get(valueID).add(featurePrtns.get(d), i, 0, 1);
						}
					} 
					catch (Exception e) {
						System.out.println("Error trying to add to a matrix");
						e.printStackTrace();
					}
				}
				
				//Check to make sure all the values were accounted for:
				for(int i = 0; i < featurePrtns.get(d).valueCount((int)splittingAtt2); i++){
					double missingVal = i;
					if(!featurePrtns2.containsKey(missingVal)){
						featurePrtns2.put(missingVal, null);
						labelPrtns2.put(missingVal, null);
					}
				}
				
				//call makeTree on each of these children
				for(Double key : labelPrtns2.keySet()){
					
					ArrayList<Double> newAttributeList = new ArrayList<Double>(attributes);

					Node newNode = new Node();
					String partOne = featurePrtns.get(d).attrValue((int) splittingAtt, d.intValue());
					String partTwo = featurePrtns.get(d).attrValue((int) splittingAtt2, key.intValue());
					newNode.setParent(partOne + partTwo);
					
					currNode.addChild(newNode);
					
					Matrix featureMatrix2 = featurePrtns2.get(key);
					
					if(featureMatrix2 != null){
						makeTree(featureMatrix2, labelPrtns2.get(key), newAttributeList, newNode);
					}
					else {
						String maxLabel = maxAttrVal(labelPrtns.get(d));
						newNode.setName(maxLabel);
						newNode.addInstances(0);
						newNode.setAttrID(labelPrtns.get(d).m_str_to_enum.get(0).get(maxLabel));
						//return; ?
					}
				}
			}			
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
	
	//Finds the maximum occurring label
	private String maxAttrVal(Matrix labels){
		
		double avLabel = labels.columnMean(0);
		double maxLabel = Math.rint(avLabel);

		return labels.m_enum_to_str.get(0).get((int)maxLabel);
	}
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		
		ArrayList<Double> attributes = new ArrayList<Double>();
		
		for(int i = 0; i < features.cols(); i++){
			attributes.add((double) i);
		}
		
		rootNode = new Node();
		makeTree(features, labels, attributes, rootNode);
//		System.out.println(rootNode.print());

	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		// TODO Auto-generated method stub
//		System.out.println("Features: ");
//		for(int i = 0; i < features.length; i++){
//			System.out.println(features[i]);
//		}
		double prediction = rootNode.classify(features);
		labels[0] = prediction;
		
	}
	
	
	
}
