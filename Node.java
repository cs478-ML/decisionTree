import java.util.ArrayList;


public class Node {
	
	private String attribute;
	private String attributeValue;
	private ArrayList<Node> children;
	private int instances;
	
	public Node(){
		attribute = "";
		children = new ArrayList<Node>();
		instances = -1;
	}
	
	public Node(String attr){
		attribute = attr;
		children = new ArrayList<Node>();
		instances = -1;
	}
	
	public void addChild(Node child){
		children.add(child);
	}
	
	public void addInstances(int i){
		instances = i;
	}
	public void setAttribute(String attr){
		attribute = attr;
	}
	public String getAttribute(){
		return attribute;
	}
	public void setAttVal(String attval){
		attributeValue = attval;
	}
	
	public String print(){
		String tree = "\n" + attributeValue + "\n" + attribute + "\n";
		for(Node n : children){
			tree = tree + "	" + n.print();
		}
		return tree;
	}

}
