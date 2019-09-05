all: pagerank 

pagerank:
	mpicc -o pagerank pagerank.c pr_graph.c -lm

clean:
	rm pagerank

