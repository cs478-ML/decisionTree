import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;


public class DecisionTree extends SupervisedLearner {
	
	private Node rootNode;
	private ArrayList < String > attrSubset;
	private Matrix origFeatures;
	private Matrix origLabels;
	
	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		origFeatures = features;
		origLabels = labels;
		attrSubset = new ArrayList< String >();
		attrSubset.addAll(features.m_attr_name);	
		rootNode = makeTree(features, labels, true);
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
				
	}
	
	/**
	 * Recursive tree construction method. 
	 * @param subsetFeatures A subset of the features Matrix which contains all
	 * the instances where a given attribute has a certain value.
	 * @param subsetLabels Same as above, but for the labels Matrix.
	 * @param infoGainCriterion A boolean value to indicate whether the splitting criterion
	 * is information gain or accuracy.
	 * @return When the recursion unfolds, the fully constructed decision tree is
	 * returned and stored in the rootNode field.
	 */
	private Node makeTree(Matrix subsetFeatures, Matrix subsetLabels, boolean infoGainCriterion) {
		if(allSameLabel(subsetLabels))
			return new LeafNode(subsetLabels.m_enum_to_str
						.get(0).get((int) subsetLabels
						.get(0, 0)));
		else if(attrSubset.size() < 2) {
			return new LeafNode(subsetLabels.m_enum_to_str
						.get(0)
						.get(Math.floor(subsetLabels.columnMean(0))));
		}
		else {
			// Here we use either information gain or accuracy to determine
			// which attribute will be used for the next decision node in
			// the tree.
			if(infoGainCriterion) {
				String bestAttr = null;
				double bestIg = 0;
				double ig = 0;
				for(String attrName : attrSubset) {
					int attrIndex = subsetFeatures.m_str_to_enum.get(0).get(attrName);
					ig = informationGain(subsetFeatures, subsetLabels, attrIndex);
					if(ig > bestIg) {
						bestIg = ig;
						bestAttr = attrName;
					}
					else if(bestAttr == null) {
						bestAttr = attrName;
					}	
				}
				attrSubset.remove(bestAttr);
				
				// TODO create the split data set and make recursive call
			}
			else {

			}
		}

		return null;
	}
	
	// Simple iterative method to check if all instances in a subset
	// have the same label.
	private boolean allSameLabel(Matrix subsetLabels) {
		double firstLabel = subsetLabels.get(0, 0);
		for(int i = 1; i < subsetLabels.rows(); i++) {
			if(subsetLabels.get(i, 0) != firstLabel)
				return false;
		}
		return true;
	}
	
	// Calculates the information gain of a given attribute in the 
	// subset of the features matrix.
	private double informationGain(Matrix subsetFeatures, Matrix subsetLabels, int attrIndex) {
		double oldEntropy = calcEntropy(subsetLabels.m_data);
		
		Map< Double, Matrix > partition = new HashMap< Double, Matrix >();
		
		for (int i = 0; i < subsetFeatures.rows(); i++) {
			if (!partition.containsKey(subsetFeatures.get(i, attrIndex))) {
				Matrix m = new Matrix(subsetLabels, i, 0, 1, 1);
				partition.put(subsetFeatures.get(i, attrIndex), m);
			}
			else {
				try {
					partition.get(subsetFeatures.get(i, attrIndex)).add(subsetLabels, i, 0, 1);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		}
		
		double newEntropy = 0;
		
		for(Double key : partition.keySet()) {
			newEntropy += ((double) partition.get(key).m_data.size())
							/ ((double) subsetLabels.rows())
							* calcEntropy(partition.get(key).m_data);
		}
		
		return oldEntropy - newEntropy;
	}
	
	// Shannon Entropy calculation method. Used in support of the 
	// informationGain method.
	private double calcEntropy(ArrayList< double[] > someLabels) {
		  Map<Double, Integer> map = new HashMap<Double, Integer>();
		  // count the occurrences of each value
		  for (double[] instance : someLabels) {
			//System.out.println(instance[0]);
		    if (!map.containsKey(instance[0])) {
		      map.put(instance[0], 0);
		    }
		    map.put(instance[0], map.get(instance[0]) + 1);
		  }
		 
		  // calculate the entropy
		  Double result = 0.0;
		  for (Double sequence : map.keySet()) {
		    Double frequency = (double) map.get(sequence) / someLabels.size();
		    result -= frequency * (Math.log(frequency) / Math.log(2));
		  }
		  
		  return result;
	}
	
	// Node classes used to build and store the decision tree model.
	abstract class Node {
		String nodeLabel;
		Node(String nl) {
			nodeLabel = nl;
		}
		String getNodeLabel() {
			return nodeLabel;
		}
	}
	
	class DecisionNode extends Node {
		TreeMap < String, Node > children;
		DecisionNode(String attrName) {
			super(attrName);
			children = new TreeMap < String, Node >();
		}
		TreeMap < String, Node > getChildren() {
			return children;
		}
	}
	
	class LeafNode extends Node {
		LeafNode(String labelVal) {
			super(labelVal);
		}
	}

}
