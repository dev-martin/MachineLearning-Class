from DT import dtree

dt  = dtree("ResultsID3.txt");

data,classes,features  = dt.read_data("carData.txt");
tree  = dt.make_tree(data,classes,features);
dt.printTree(tree,'', True);

output = open("ResultsID3.txt", "a")

output.write("Number of training samples = " + str(len(data))+ "\n" )

results = dt.classifyAll(tree,data)
output.write("Number of ACCURATE CLASSIFIED training samples = " + str(len(results)) + "\n")

output.write("          Martin Iglesias      ~       116928472 ")

