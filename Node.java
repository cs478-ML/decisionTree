import java.util.ArrayList;


public class Node {
	
	private String attribute;
	private ArrayList<Node> children;
	
	public Node(String attr){
		attribute = attr;
		children = new ArrayList<Node>();
	}
	
	public void addChild(Node child){
		children.add(child);
	}

}
