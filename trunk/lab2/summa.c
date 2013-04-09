/**
 *  \file summa.c
 *  \desc Implements a distributed 2D SUMMA matrix multiply algorithm.
 */

#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include "mpi_helper.h"
#include "mat.h"
#include "mm1d.h"
#include "summa.h"

#define MSGTAG_SUMMA_DIST 101

/* ------------------------------------------------------------ */

/**
 *  \brief Copies an mb x nb block of A to B.
 *
 *  Both A and B are stored in column-major order, with the leading
 *  dimension of A being lda.
 */
static
void
copyBlock__ (int mb, int nb, double* B, const double* A, int lda)
{
  for (int j = 0; j < nb; ++j) /* loop over columns */
    memcpy (B + j*mb, A + j*lda, mb * sizeof (double));
}

/* ------------------------------------------------------------ */

MPI_Comm
summa_createTopology (MPI_Comm comm, int Pr, int Pc)
{
  const int dims[2] = {Pr, Pc};
  const int periods[2] = {0, 0};
  const int reorder = 1;
  MPI_Comm comm2d;
  MPI_Cart_create (comm, 2, (int *)dims, (int *)periods, reorder, &comm2d);
  return comm2d;
}

void
summa_freeTopology (MPI_Comm comm)
{
  MPI_Comm_free (&comm);
}

void
summa_getProcCoords (MPI_Comm comm2d, int* p_rank_row, int* p_rank_col)
{
  int dims[2];
  int periods[2];
  int coords[2];
  MPI_Cart_get (comm2d, 2, dims, periods, coords);
  if (p_rank_row) *p_rank_row = coords[0];
  if (p_rank_col) *p_rank_col = coords[1];
}

void
summa_getProcSize (MPI_Comm comm2d, int* p_P_row, int* p_P_col)
{
  int dims[2];
  int periods[2];
  int coords[2];
  MPI_Cart_get (comm2d, 2, dims, periods, coords);
  if (p_P_row) *p_P_row = dims[0];
  if (p_P_col) *p_P_col = dims[1];
}

void
summa_getMatDims (int m, int n, MPI_Comm comm2d,
		  int* p_m_local, int* p_n_local)
{
  int P_row, P_col;
  summa_getProcSize (comm2d, &P_row, &P_col);
  int rank_row, rank_col;
  summa_getProcCoords (comm2d, &rank_row, &rank_col);
  int m_local = mm1d_getBlockLength (m, P_row, rank_row);
  int n_local = mm1d_getBlockLength (n, P_col, rank_col);
  if (p_m_local) *p_m_local = m_local;
  if (p_n_local) *p_n_local = n_local;
}

/* ------------------------------------------------------------ */

double *
summa_distribute (int m, int n, const double* A, int root, MPI_Comm comm2d)
{
  int rank = mpih_getRank (comm2d);
  int P_row, P_col;
  summa_getProcSize (comm2d, &P_row, &P_col);
  int rank_row, rank_col;
  summa_getProcCoords (comm2d, &rank_row, &rank_col);

  int m_local = mm1d_getBlockLength (m, P_row, rank_row);
  int n_local = mm1d_getBlockLength (n, P_col, rank_col);
  double* A_local = (double *)malloc (m_local * n_local * sizeof (double));
  mpih_assert (A_local != NULL);

  if (rank == root) { /* Owner of A */
    MPI_Request* reqs = (MPI_Request *)malloc (P_row * P_col * sizeof (MPI_Request));
    int n_req = 0;
    /* Send submatrix to the appropriate process */
    for (int rI = 0; rI < P_row; ++rI) {
      for (int rJ = 0; rJ < P_col; ++rJ) {
	if (rI == 0 && rJ == 0) { /* root; make local copy */
	  copyBlock__ (m_local, n_local, A_local, A, m);
	} else {
	  int i0 = mm1d_getBlockStart (m, P_row, rI);
	  int m_dest = mm1d_getBlockLength (m, P_row, rI);
	  int j0 = mm1d_getBlockStart (n, P_col, rJ);
	  int n_dest = mm1d_getBlockLength (n, P_col, rJ);

	  int coords_dest[2] = {rI, rJ};
	  int rank_dest;
	  MPI_Cart_rank (comm2d, coords_dest, &rank_dest);

	  MPI_Datatype block_t;
	  MPI_Type_vector (n_dest, m_dest, m, MPI_DOUBLE, &block_t);
	  MPI_Type_commit (&block_t);
	  MPI_Isend ((double *)A + i0 + j0*m, 1, block_t, rank_dest,
		     MSGTAG_SUMMA_DIST, comm2d, &(reqs[n_req++]));
	  MPI_Type_free (&block_t);
	}
      } /* rJ loop */
    } /* rI loop */
    MPI_Waitall (n_req, reqs, MPI_STATUSES_IGNORE);
  }
  else { /* rank != root */
    MPI_Status stat;
    MPI_Recv (A_local, m_local * n_local, MPI_DOUBLE, 0, MSGTAG_SUMMA_DIST,
	      comm2d, &stat);
  }
  return A_local;
}

/* ------------------------------------------------------------ */

/**
 *  \brief Given a communicator for a 2D process grid, this routine
 *  returns a new communicator consisting only of the process-grid row
 *  in which the calling process belongs.
 *
 *  Example: If the process grid is 2 x 3, e.g.,
 *
 *     (0, 0)  |  (0, 1)  |  (0, 2)
 *     (1, 0)  |  (1, 1)  |  (1, 2)
 *
 *  and process (1, 1) calls this routine, then the routine will
 *  return a communicator containing the processes, {(1, 0), (1, 1),
 *  (1, 2)}.
 */
static
MPI_Comm
getCommRow__ (MPI_Comm comm2d)
{
  int select[2] = {0, 1};
  MPI_Comm comm_row;
  MPI_Cart_sub (comm2d, select, &comm_row);
  return comm_row;
}

/**
 *  \brief Given a communicator for a 2D process grid, this routine
 *  returns a new communicator consisting only of the process-grid
 *  column in which the calling process belongs.
 *
 *  Example: If the process grid is 2 x 3, e.g.,
 *
 *     (0, 0)  |  (0, 1)  |  (0, 2)
 *     (1, 0)  |  (1, 1)  |  (1, 2)
 *
 *  and process (1, 1) calls this routine, then the routine will
 *  return a communicator containing the processes, {(0, 0), (1, 1)}.
 */
static
MPI_Comm
getCommCol__ (MPI_Comm comm2d)
{
  int select[2] = {1, 0};
  MPI_Comm comm_col;
  MPI_Cart_sub (comm2d, select, &comm_col);
  return comm_col;
}

/* ------------------------------------------------------------ */

void
summa_mult (int m, int n, int k, int s_max,
	    const double* A_local, const double* B_local,
	    double* C_local, MPI_Comm comm2d,
	    double* p_t_comp, double* p_t_comm)
{
  int rank = mpih_getRank (comm2d);
  int P_row, P_col;
  summa_getProcSize (comm2d, &P_row, &P_col);
  int rank_row, rank_col;
  summa_getProcCoords (comm2d, &rank_row, &rank_col);
  MPI_Comm comm_row = getCommRow__ (comm2d);
  MPI_Comm comm_col = getCommCol__ (comm2d);

  int m_local, n_local;
  summa_getMatDims (m, n, comm2d, &m_local, &n_local);

  double* A_strip = (double *)malloc (m_local * s_max * sizeof (double));
  double* B_strip = (double *)malloc (s_max * n_local * sizeof (double));
  mpih_assert (A_strip && B_strip);

  double t_comp = 0, t_comm = 0; /* Time in computation vs. communication */
  int iter_k = 0; /* iterates over strips in k dimension */
  int owner_A = 0; /* owner of current A strip */
  int owner_B = 0; /* owner of current B strip */
  while (iter_k < k) {
    int k_start_A = mm1d_getBlockStart (k, P_col, owner_A);
    int k_local_A = mm1d_getBlockLength (k, P_col, owner_A);

    int k_start_B = mm1d_getBlockStart (k, P_row, owner_B);
    int k_local_B = mm1d_getBlockLength (k, P_row, owner_B);

    /* Determine a strip width, s, that still resides in both the
       current A and B blocks. */
    int s = min_int (s_max, min_int (k_start_A + k_local_A - iter_k,
				     k_start_B + k_local_B - iter_k));

    double t_start = MPI_Wtime (); /* For timing communication */
    /* Step 1: Broadcast m_local x s strip of A */
    /* === @@ YOUR CODE GOES HERE @@ === */
    /* Step 2: Broadcast s x n_local strip of B */
    /* === @@ YOUR CODE GOES HERE @@ === */
    t_comm += MPI_Wtime () - t_start;

    /* Step 3: Local multiply */
    t_start = MPI_Wtime (); /* For timing computation */
    mat_multiply (m_local, n_local, s,
		  A_strip, m_local, B_strip, s,
		  C_local, m_local);
    t_comp += MPI_Wtime () - t_start;

    iter_k += s;
    if (iter_k >= k_start_A + k_local_A) ++owner_A;
    if (iter_k >= k_start_B + k_local_B) ++owner_B;
  }

  /* Clean-up and return */
  MPI_Comm_free (&comm_row);
  MPI_Comm_free (&comm_col);
  free (A_strip);
  free (B_strip);
  if (p_t_comp) *p_t_comp = t_comp;
  if (p_t_comm) *p_t_comm = t_comm;
}

/* ------------------------------------------------------------ */

double *
summa_alloc (int m, int n, MPI_Comm comm2d)
{
  int m_local, n_local;
  summa_getMatDims (m, n, comm2d, &m_local, &n_local);
  double* A_local = (double *)malloc (m_local * n_local * sizeof (double));
  mpih_assert (A_local != NULL);
  return A_local;
}

void
summa_free (double* A_local, MPI_Comm comm2d)
{
  if (A_local) free (A_local);
}

void
summa_randomize (int m, int n, double* A_local, MPI_Comm comm2d)
{
  int m_local, n_local;
  summa_getMatDims (m, n, comm2d, &m_local, &n_local);
  mpih_assert (A_local || !m_local || !n_local);
  for (int i = 0; i < m_local; ++i)
    for (int j = 0; j < n_local; ++j)
      A_local[i + j*m_local] = drand48 ();
}

void
summa_setZero (int m, int n, double* A_local, MPI_Comm comm2d)
{
  int m_local, n_local;
  summa_getMatDims (m, n, comm2d, &m_local, &n_local);
  mpih_assert (A_local || !m_local || !n_local);
  if (A_local)
    bzero (A_local, m_local * n_local * sizeof (double));
}

/* ------------------------------------------------------------ */

void
summa_dump (const char* tag, int m, int n, const double* A_local, MPI_Comm comm2d)
{
  int P_row, P_col;
  summa_getProcSize (comm2d, &P_row, &P_col);
  int rank = mpih_getRank (comm2d);
  int rank_row, rank_col;
  summa_getProcCoords (comm2d, &rank_row, &rank_col);

  int m_local = mm1d_getBlockLength (m, P_row, rank_row);
  int n_local = mm1d_getBlockLength (n, P_col, rank_col);
  int i0 = mm1d_getBlockStart (m, P_row, rank_row);
  int j0 = mm1d_getBlockStart (n, P_col, rank_col);

  for (int r_row = 0; r_row < P_row; ++r_row) {
    for (int r_col = 0; r_col < P_col; ++r_col) {
      MPI_Barrier (comm2d); /* Serialize output */
      if (r_row == rank_row && r_col == rank_col) {
	fflush (stderr);
	mpih_debugmsg (comm2d, "(p%d,p%d)>> Mat:%s -- %d x %d; j \\in [%d,%d]\n",
		       rank_row, rank_col, tag, m_local, n_local,
		       j0, j0+n_local-1);

	/* Maximum no. of rows/columns to print */
	const int MAX_ROWS = 4;
	const int MAX_COLS = 4;

	for (int di = 0; di < min_int (m_local, MAX_ROWS); ++di) {
	  fprintf (stderr, "   Row %d:", i0+di);
	  for (int dj = 0; dj < min_int (n_local, MAX_COLS); ++dj) {
	    fprintf (stderr, " %g", A_local[di + dj*m_local]);
	  } /* dj */
	  if (n_local > MAX_COLS) fprintf (stderr, " ... (omitted)");
	  fprintf (stderr, "\n");
	} /* di */
	if (m_local > MAX_ROWS) fprintf (stderr, "   ... (rows omitted) ...\n");
      } /* (r_row, r_col) == (rank_row, rank_col) */
    } /* r_col */
  } /* r_row */
}

/* eof */
