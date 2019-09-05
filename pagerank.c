
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

#include "pr_graph.h"



/**
 * @brief Compute the PageRank (PR) of a graph.
 *
 * @param graph The graph.
 * @param damping Damping factor (or, 1-restart). 0.85 is typical.
 * @param max_iterations The maximium number of iterations to perform.
 *
 * @return A vector of PR values.
 */
double * pagerank(
        pr_graph const * const graph,
        double const damping,
        int const max_iterations, int n, int p, int rank, int *sendcnts, int *recvcnts, int *displs);

int getProcess(int vertex, int *displs, int p){
    for(int k = 0; k < p; k++)
        if(vertex >= displs[k] && vertex < displs[k + 1])
            return k;
}
static inline double monotonic_seconds(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
int main(
        int argc,
        char * * argv)
{
    if(argc == 1) {
        fprintf(stderr, "usage: %s <graph> [output file]\n", *argv);
        return EXIT_FAILURE;
    }
    int p, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char * ifname = argv[1];
    char * ofname = NULL;
    if(argc > 2) {
        ofname = argv[2];
    }
    pr_graph * graph;

    graph = malloc(sizeof(*graph));
    int no_verts, no_edges, *displs, *sendcnts, *recvcnts, n;
    displs = (int*)malloc((p + 1) * sizeof(int));
    sendcnts = (int*)malloc(p * sizeof(int));
    recvcnts = (int*)malloc(p * sizeof(int));
    if(rank == p - 1){
        graph = pr_graph_load(ifname, p, displs);
        no_verts = graph->nvtxs;
        no_edges = graph->xadj[no_verts];
    }
    else{
        MPI_Status status;
        MPI_Recv(&no_verts, 1, MPI_INT, p - 1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&no_edges, 1, MPI_INT, p - 1, 1, MPI_COMM_WORLD, &status);
        graph->xadj = malloc((no_verts + 1) * sizeof(*graph->xadj));
        graph->nbrs = malloc(no_edges * sizeof(*graph->nbrs));
        MPI_Recv(graph->xadj, no_verts + 1, MPI_UINT64_T, p - 1, 2, MPI_COMM_WORLD, &status);
        MPI_Recv(graph->nbrs, no_edges, MPI_UINT64_T, p - 1, 3, MPI_COMM_WORLD, &status);
        graph->nvtxs = no_verts;
    }
    MPI_Bcast(displs, p + 1, MPI_INT, p - 1, MPI_COMM_WORLD);
    for(int i = 0; i < p; i++)sendcnts[i] = 0;
    int ptr = 0;
    for(int i = 0; i < no_verts; i++){
        /* qsort(&graph->nbrs[ptr], graph->xadj[i + 1] - graph->xadj[i], sizeof(uint64_t), cmpfunc); */
        for(int j = 0; j < graph->xadj[i + 1] - graph->xadj[i]; j++){
            int vertex = graph->nbrs[ptr++];
            sendcnts[getProcess(vertex, displs, p)]++;
        }
    }
    MPI_Alltoall(sendcnts, 1, MPI_INT, recvcnts, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allreduce(&no_verts, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    /* if(!graph) { */
    /*     return EXIT_FAILURE; */
    /* } */
    double time1 = monotonic_seconds();
    double * PR = pagerank(graph, 0.85, 10, n, p, rank, sendcnts, recvcnts, displs);
    double time2 = monotonic_seconds();
    if(rank == 0)printf("Number of iterations: 10 average time: %0.04fs\n", (time2 - time1) / (double)100);
    if(rank != 0){
        if(ofname)MPI_Send(PR, no_verts, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else{
        double *toWrite;
        if(ofname) {
            FILE * fout = fopen(ofname, "w");
            if(!fout) {
                fprintf(stderr, "ERROR: could not open '%s' for writing.\n", ofname);
                return EXIT_FAILURE;
            }
            for(pr_int v = 0; v < graph->nvtxs; v++) {
                fprintf(fout, "%0.3e\n", PR[v]);
            }
            int ubVertices = (n + p - 1) / p;
            toWrite = (double*)malloc(ubVertices * sizeof(double));
            MPI_Status status;
            for(int i = 1; i < p; i++){
                int no = displs[i + 1] - displs[i];
                MPI_Recv(toWrite, no, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
                for(int j = 0; j < no; j++)
                    fprintf(fout, "%0.3e\n", toWrite[j]);
            }
            fclose(fout);
        }
    }
    MPI_Finalize();
    return EXIT_SUCCESS;
}



double * pagerank(
        pr_graph const * const graph,
        double const damping,
        int const max_iterations, int n, int p, int rank, int *sendcnts, int *recvcnts, int *displs)
{
    /* grab graph structures to save typing */
    pr_int const nvtxs = graph->nvtxs;
    pr_int const * const restrict xadj = graph->xadj;
    pr_int const * const restrict nbrs = graph->nbrs;

    /* Initialize pageranks to be a probability distribution. */
    double * PR = malloc(nvtxs * sizeof(*PR));
    for(pr_int v=0; v < nvtxs; ++v) {
        PR[v] = 1. / (double) n;
    }

    /* Probability of restart */
    double const restart = (1 - damping) / (double) n;


    /* Convergence tolerance. */
    double const tol = 1e-9;

    double * PR_accum = malloc(nvtxs * sizeof(*PR));
   
    /* Allocate the arrays that need to be sent and received */
    /* Allocate displacement arrays for both send and receive. */
    
    int toSend = 0, toRecv = 0, *sDispls, *rDispls, *ptr, *sVertex, *rVertex;
    for(int i = 0; i < p; i++)toSend += sendcnts[i];
    for(int i = 0; i < p; i++)toRecv += recvcnts[i];

    double *sendArray, *recvArray;

    sendArray = (double*)malloc(toSend * sizeof(double));
    recvArray = (double*)malloc(toRecv * sizeof(double));
    sVertex = (int*)malloc(toSend * sizeof(int));
    rVertex = (int*)malloc(toRecv * sizeof(int));
   
    sDispls = (int*)malloc(p * sizeof(int));
    rDispls = (int*)malloc(p * sizeof(int));
    ptr = (int*)malloc(p * sizeof(int));

    sDispls[0] = 0;
    for(int i = 1; i < p; i++)sDispls[i] = sDispls[i - 1] + sendcnts[i - 1];

    rDispls[0] = 0;
    for(int i = 1; i < p; i++)rDispls[i] = rDispls[i - 1] + recvcnts[i - 1];
    
    for(int currIter = 0; currIter < max_iterations; currIter++){
        for(int i = 0; i < p; i++)ptr[i] = sDispls[i];

        int lptr = 0;
        for(int i = 0; i < graph->nvtxs; i++){
            for(int j = 0; j < graph->xadj[i + 1] - graph->xadj[i]; j++){
                int vertex = graph->nbrs[lptr++];
                int num_links = graph->xadj[i + 1] - graph->xadj[i];
                int processToSend = getProcess(vertex, displs, p);
                sendArray[ptr[processToSend]] = (PR[i] / (double)num_links);
                sVertex[ptr[processToSend]++] = vertex - displs[processToSend];
            }
        }
        MPI_Alltoallv(sendArray, sendcnts, sDispls, MPI_DOUBLE, recvArray, recvcnts, rDispls, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Alltoallv(sVertex, sendcnts, sDispls, MPI_INT, rVertex, recvcnts, rDispls, MPI_INT, MPI_COMM_WORLD);
        for(int i = 0; i < graph->nvtxs; i++)
            PR_accum[i] = 0.0;
        for(int i = 0; i < toRecv; i++)
            PR_accum[rVertex[i]] += recvArray[i];
        for(int i = 0; i < graph->nvtxs; i++)
            PR_accum[i] = (PR_accum[i] * damping) + restart;
        double norm = 0.0, sumNorm = 0.0;
        for(int i = 0; i < graph->nvtxs; i++)
            norm += (PR[i] - PR_accum[i]) * (PR[i] - PR_accum[i]);
        for(int i = 0; i < graph->nvtxs; i++)
            PR[i] = PR_accum[i];
        MPI_Allreduce(&norm, &sumNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        sumNorm = sqrt(sumNorm);
        if(currIter > 1 && sumNorm < tol){
            break;
        }
    }
    free(PR_accum);
    return PR;
}
