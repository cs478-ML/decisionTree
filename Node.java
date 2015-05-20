import java.util.ArrayList;


public class Node {
	
	private String name;
	private String parent;
	private ArrayList<Node> children;
	private int instances;
	
	private double attrID;
	private double attrID2 = -1;
	
	public Node(){
		name = "";
		children = new ArrayList<Node>();
		instances = -1;
	}
	
	public Node(String attr){
		name = attr;
		children = new ArrayList<Node>();
		instances = -1;
	}
	
	public void addChild(Node child){
		children.add(child);
	}
	
	public void addInstances(int i){
		instances = i;
	}
	public int getInstances(){
		return instances;
	}
	public void setName(String attr){
		name = attr;
	}
	public String getName(){
		return name;
	}
	public void setParent(String attval){
		parent = attval;
	}
	
	public String print(){
		String tree = "\n" + parent + "\n" + name + "(" + attrID + ")\n		instances: " + instances;
		for(Node n : children){
			tree = tree + "	" + n.print();
		}
		return tree;
	}
	
	public void printNode(){
		String tree = "\n" + name + "(" + attrID + ")\n		instances: " + instances;
		for(Node n : children){
			tree = tree + "	" + n.name;
		}
		
		System.out.println(tree);
	}
	
	public void populate (String attr, int i, double id) {
		name = attr;
		instances = i;
		attrID = id;
	}
	
	public void setAttrID(double id){
		attrID = id;
	}
	
	public void setAttrID(double attr1, double attr2){
		attrID = attr1;
		attrID2 = attr2;
	}
	
	public double classify(double[] sample){
		
		//printNode();
		
		if(instances != -1){
			return attrID;
		}
		
		double nextNodeIndex = sample[(int)attrID];
		
		if (nextNodeIndex > children.size() - 1) {
			nextNodeIndex = children.size() - 1;
		}
		if (children.size() == 0)
			return attrID;
		if (nextNodeIndex == -1) {
			System.out.println("-1");
		}
		return children.get((int)nextNodeIndex).classify(sample);
	}

}
