/**
 *  \file summa.h
 *  \desc Implements a distributed 2D SUMMA matrix multiply algorithm.
 */

#if !defined (INC_SUMMA_H)
#define INC_SUMMA_H /*!< summa.h included */

#include <mpi.h>
#include "util.h"

/** Creates a 2D process grid */
MPI_Comm summa_createTopology (MPI_Comm comm, int Pr, int Pc);

/** Release resources for previously created 2D process grid. */
void summa_freeTopology (MPI_Comm comm);

/** Returns the process-grid coordinates of the calling process. */
void summa_getProcCoords (MPI_Comm comm2d, int* p_rank_row, int* p_rank_col);

/** Returns the process-grid dimensions. */
void summa_getProcSize (MPI_Comm comm2d, int* p_P_row, int* p_P_col);

/** Returns the local matrix dimensions for an m x n matrix
 *  distributed in comm2d.
 */
void
summa_getMatDims (int m, int n, MPI_Comm comm2d,
		  int* p_m_local, int* p_n_local);

/**
 *  \brief Given an m x n matrix A stored on process root, this
 *  collective routine distributes blocks of A among the Pr x Pc
 *  processors in comm, returning a pointer to the local block.
 */
double* summa_distribute (int m, int n, const double* A,
			  int root, MPI_Comm comm2d);

/**
 *  \brief Performs a distributed matrix multiply using the 2D SUMMA
 *  algorithm.
 *
 *  A is m x n, B is k x n, and C is m x n. All three matrices are
 *  distributed across the processors in comm in blocks.
 */
void summa_mult (int m, int n, int k, int s_max,
		 const double* A_local, const double* B_local,
		 double* C_local, MPI_Comm comm2d,
		 double* p_t_comp, double* p_t_comm, int type);

/**
 * \brief Allocates a M x N matrix across all processes in comm using
 * a 1D block column partitioning, returning a pointer to the local
 * block.
 */
double* summa_alloc (int m, int n, MPI_Comm comm2d);

/** \brief Frees local storage */
void summa_free (double* A_local, MPI_Comm comm2d);

/** \brief Sets matrix entries to random values in [0, 1]. */
void summa_randomize (int m, int n, double* A_local, MPI_Comm comm2d);

/** \brief Sets matrix entries to 0. */
void summa_setZero (int m, int n, double* A_local, MPI_Comm comm2d);

/** \brief (Debug) Print the matrix. */
void summa_dump (const char* tag, int m, int n, const double* A_local,
		 MPI_Comm comm2d);

#endif

/* eof */
