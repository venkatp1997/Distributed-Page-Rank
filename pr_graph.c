

/* ensure we have `getline()` */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>

#include "pr_graph.h"



pr_graph * pr_graph_load(
        char const * const ifname, int p, int *displs)
{
    FILE * fin = fopen(ifname, "r");
    if(!fin) {
        fprintf(stderr, "ERROR: could not open '%s' for reading.\n", ifname);
        return NULL;
    }

    pr_graph * graph = malloc(sizeof(*graph));

    /* read nvtxs and nedges */
    fscanf(fin, "%lu", &(graph->nvtxs));
    fscanf(fin, "%lu", &(graph->nedges));
    fscanf(fin, "\n"); /* make sure we process the newline, too. */
    
    int n = graph->nvtxs;
    int no_verts = (n + p - 1) / p;
    int currDone = 0, count = 0, sz = 0, currProcess = 0;
    displs[0] = 0;

    graph->xadj = malloc((no_verts + 1) * sizeof(*graph->xadj));
    /* graph->nbrs = malloc(graph->nedges * sizeof(*graph->nbrs)); */
    pr_int *tmp_edges = NULL;
    graph->nbrs = tmp_edges;

    /* How many edges we have read. */
    pr_int edge_ptr = 0;

    char * line = malloc(1024 * 1024);
    size_t len = 0;

    /* Read in graph one vertex at a time. */
    int bcast = 1;
    for(pr_int v=0; v < n; ++v) {
        ssize_t read = getline(&line, &len, fin);
        if(read == -1) {
            fprintf(stderr, "ERROR: premature EOF at line %lu\n", v+1);
            pr_graph_free(graph);
            return NULL;
        }

        /* Store the beginning of the adjacency list. */
        graph->xadj[currDone] = edge_ptr;

        /* Check for sinks -- these make pagerank more difficult. */
        if(read == 1) {
            fprintf(stderr, "WARNING: vertex '%lu' is a sink vertex.\n", v+1);
            continue;
        }

        /* Foreach edge in line. */
        char * ptr = strtok(line, " ");
        while(ptr != NULL) {
            char * end = NULL;
            pr_int const e_id = strtoull(ptr, &end, 10);
            /* end of line */
            if(ptr == end) {
                break;
            }
            /* assert(e_id > 0 && e_id <= graph->nvtxs); */
            count++;
            if(count > sz){
                if(sz == 0){
                    tmp_edges = realloc(tmp_edges, sizeof(*tmp_edges));
                    sz = 1;
                }
                else{
                    tmp_edges = realloc(tmp_edges, 2 * sz * sizeof(*tmp_edges));
                    sz *= 2;
                }
                graph->nbrs = tmp_edges;
            }
            graph->nbrs[edge_ptr++] = e_id - 1; /* 1 indexed */
            ptr = strtok(NULL, " ");
        }
        currDone++;
        if(currDone == no_verts || v == n - 1){
            graph->nvtxs = currDone;
            graph->xadj[currDone] = count;
            tmp_edges = realloc(tmp_edges, count * sizeof(*tmp_edges));
            graph->nbrs = tmp_edges;
            graph->xadj = realloc(graph->xadj, (currDone + 1) * sizeof(*graph->xadj));
            displs[currProcess + 1] = displs[currProcess] + currDone;
            if(currProcess != p - 1){
                MPI_Send(&no_verts, 1, MPI_INT, currProcess, 0, MPI_COMM_WORLD);
                MPI_Send(&edge_ptr, 1, MPI_INT, currProcess, 1, MPI_COMM_WORLD);
                MPI_Send(graph->xadj, currDone + 1, MPI_UINT64_T, currProcess, 2, MPI_COMM_WORLD);
                MPI_Send(graph->nbrs, count, MPI_UINT64_T, currProcess, 3, MPI_COMM_WORLD);

                currProcess++;
                currDone = 0;
                count = 0;
                sz = 0;
                edge_ptr = 0;

                free(graph->xadj);
                free(tmp_edges);
                tmp_edges = NULL;
                graph->nbrs = tmp_edges;
                graph->xadj = malloc((no_verts + 1) * sizeof(*graph->xadj));
            }
            else{
                return graph;
            }
        }
    }
    /* assert(edge_ptr == graph->nedges); */
    /* graph->xadj[graph->nvtxs] = graph->nedges; */

    /* free(line); */

    /* return graph; */
}


void pr_graph_free(
        pr_graph * const graph)
{
    free(graph->xadj);
    free(graph->nbrs);
    free(graph);
}


