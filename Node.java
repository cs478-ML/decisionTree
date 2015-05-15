import java.util.ArrayList;


public class Node {
	
	private String name;
	private String parent;
	private ArrayList<Node> children;
	private int instances;
	
	private double attrID;
	
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
	public void setAttrID(double id){
		attrID = id;
	}
	public double getAttrID(){
		return attrID;
	}
	
	public double classify(double[] sample){
		
		printNode();
		
		if(instances != -1){
			return attrID;
		}
		else {
			double nextNodeIndex = sample[(int)attrID];
			return children.get((int)nextNodeIndex).classify(sample);
		}
	}

}
