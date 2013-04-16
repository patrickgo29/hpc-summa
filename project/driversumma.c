/*
 *  \file driversumma.c
 *
 *  \brief Driver program for a distributed 2D blocked matrix multiply
 *  timing/testing program, based on the SUMMA algorithm.
 *
 *  Adapted from code by Jason Riedy, David Bindel, David Garmire,
 *  Kent Czechowski, Aparna Chandramowlishwaran, and Richard Vuduc.
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <float.h>
#include <math.h>

#include "util.h"

#include <mpi.h>
#include "mpi_helper.h"

#include "mat.h" // sequential algorithm
#include "mm1d.h" // 1D algorithm
#include "summa.h" // SUMMA algorithm

/* ------------------------------------------------------------ */

/** Prints help message */
static void usage__ (const char* progname);

/** \brief Checks the distributed matrix multiply routine */
static void verify__ (int m, int n, int k, int P_row, int P_col, int s);

/**
 *  \brief Print aggregate execution time statistics for each of the
 *  given measurements 't[0..n_t-1]' on the local processor.
 *
 *  \note Set 'debug' to a non-zero value to print to stderr instead
 *  of stdout.
 */
static void summarize__ (int m, int n, int k, int s,
			 const double* t, int n_t,
			 MPI_Comm comm, int debug);

/** \brief Checks the distributed matrix multiply routine */
static void benchmark__ (int m, int n, int k, int P_row, int P_col, int s);

/* ------------------------------------------------------------ */

/** Program starts here */
int
main (int argc, char** argv)
{
  int retcode = MPI_Init (&argc, &argv);
  mpih_assert (retcode == MPI_SUCCESS);

  int rank = mpih_getRank (MPI_COMM_WORLD);
  int P = mpih_getSize (MPI_COMM_WORLD);

  srand48 ((long)rank);

  int M, N, K; /* matrix dimensions */
  int Pr, Pc; /* process grid */
  int strip_width; /* strip width */
  if (rank == 0) { /* p0 parses the command-line arguments */
    if (argc != 7) {
      usage__ (argv[0]);
      MPI_Abort (MPI_COMM_WORLD, 1);
    }
    M = atoi (argv[1]);  mpih_assert (M > 0);
    N = atoi (argv[2]);  mpih_assert (N > 0);
    K = atoi (argv[3]);  mpih_assert (K > 0);
    Pr = atoi (argv[4]); mpih_assert (Pr > 0);
    Pc = atoi (argv[5]); mpih_assert (Pc > 0);
    strip_width = atoi (argv[6]); mpih_assert (strip_width > 0);
    mpih_assert ((Pr * Pc) == P);
  }

  /* p0 then distributes the program arguments */
  MPI_Bcast (&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast (&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast (&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast (&Pr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast (&Pc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast (&strip_width, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    mpih_debugmsg (MPI_COMM_WORLD, "Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    mpih_debugmsg (MPI_COMM_WORLD, "Process grid: %d x %d\n", Pr, Pc);
    mpih_debugmsg (MPI_COMM_WORLD, "SUMMA strip width: %d\n", strip_width);
  }

  verify__ (M, N, K, Pr, Pc, strip_width);
  benchmark__ (M, N, K, Pr, Pc, strip_width);

  MPI_Finalize ();
  return 0;
}

static
void
usage__ (const char* progname)
{
  fprintf (stderr, "\n");
  fprintf (stderr, "usage: %s <m> <n> <k> <Pr> <Pc> <s>\n", progname);
  fprintf (stderr, "\n");
  fprintf (stderr,
	   "Performs C <- C + A*B using the 2D SUMMA algorithm with strip-width s on a Pr x Pc process grid.\n");
  fprintf (stderr, "\n");
}

/* ------------------------------------------------------------ */

/**
 *  \brief Creates a sequential baseline matrix multiply problem
 *  instance.
 */
static void setupSeqProblem__ (int m, int n, int k,
			       double** p_A, double** p_B,
			       double** p_C, double** p_C_bound);

/**
 *  \brief Returns a 1 if the envirionment variable var is enabled,
 *  i.e., set to 'yes' or 'y' or any integer value >= 1.
 */
static int checkEnvEnabled__ (const char* var, int def_val);

static
void
verify__ (int m, int n, int k, int P_row, int P_col, int s)
{
  if (!checkEnvEnabled__ ("VERIFY", 1)) return;

  MPI_Comm comm2d = summa_createTopology (MPI_COMM_WORLD, P_row, P_col);
  int rank = mpih_getRank (comm2d);

  double* A = NULL;
  double* B = NULL;
  double* C_soln = NULL;
  double* C_bound = NULL;

  /* Whoever has rank == 0 will create the test problem. */
  if (rank == 0) {
    setupSeqProblem__ (m, n, k, &A, &B, &C_soln, &C_bound);

    /* Measure time for the sequential problem. */
    mat_setZero (m, n, C_soln);
    double t_start = MPI_Wtime ();
    mat_multiply (m, n, k, A, m, B, k, C_soln, m);
    double dt_seq = MPI_Wtime () - t_start;
    mpih_debugmsg (MPI_COMM_WORLD, "t_seq = %g s\n", dt_seq);

    /* Recompute, to get the error bound this time */
    mpih_debugmsg (MPI_COMM_WORLD, "Estimating error bound...\n");
    mat_multiplyErrorbound (m, n, k, A, m, B, k, C_soln, m, C_bound, m);
  }

  /* Next, run the (untrusted) SUMMA algorithm */
  if (rank == 0) mpih_debugmsg (comm2d, "Distributing A, B, and C...\n");
  double* A_local = summa_distribute (m, k, A, 0, comm2d);
  double* B_local = summa_distribute (k, n, B, 0, comm2d);
  double* C_local = summa_alloc (m, n, comm2d);
  summa_setZero (m, n, C_local, comm2d);

  /* Do multiply */
  if (rank == 0) mpih_debugmsg (comm2d, "Computing C <- C + A*B...\n");
  summa_mult (m, n, k, s, A_local, B_local, C_local, comm2d, NULL, NULL);

  /* Compare the two answers (in parallel) */
  if (rank == 0) mpih_debugmsg (comm2d, "Verifying...\n");
  int rank_row, rank_col;
  summa_getProcCoords (comm2d, &rank_row, &rank_col);
  double* C_soln_local = summa_distribute (m, n, C_soln, 0, comm2d);
  double* C_bound_local = summa_distribute (m, n, C_bound, 0, comm2d);
  int m_local = mm1d_getBlockLength (m, P_row, rank_row);
  int n_local = mm1d_getBlockLength (n, P_col, rank_col);
  for (int i = 0; i < m_local; ++i) {
    for (int j = 0; j < n_local; ++j) {
      const double errbound = C_bound_local[i + j*m_local] * 3.0 * k * DBL_EPSILON;
      const double c_trusted = C_soln_local[i + j*m_local]; 
      const double c_untrusted = C_local[i + j*m_local];
      double delta = fabs (c_untrusted - c_trusted);
      if (delta > errbound)
	mpih_debugmsg (comm2d,
		       "*** Entry (%d, %d) --- Error bound violated ***\n    ==> |%g - %g| == %g > %g\n",
		       c_untrusted, c_trusted, delta, errbound, i, j);
      mpih_assert (delta <= errbound);
    }
  }
  if (rank == 0) mpih_debugmsg (comm2d, "Passed!\n");

  /* Clean-up */
  summa_free (A_local, comm2d);
  summa_free (B_local, comm2d);
  summa_free (C_local, comm2d);
  summa_free (C_soln_local, comm2d);
  summa_free (C_bound_local, comm2d);
  if (rank == 0) {
    free (A);
    free (B);
    free (C_soln);
    free (C_bound);
  }
  summa_freeTopology (comm2d);
}

/* ------------------------------------------------------------ */

static
void
setupSeqProblem__ (int m, int n, int k,
		   double** p_A, double** p_B,
		   double** p_C, double** p_C_bound)
{
  if (p_A) {
    *p_A = mat_create (m, k);
    mat_randomize (m, k, *p_A);
  }
  if (p_B) {
    *p_B = mat_create (k, n);
    mat_randomize (k, n, *p_B);
  }
  if (p_C) {
    *p_C = mat_create (m, n);
    mat_setZero (m, n, *p_C);
  }
  if (p_C_bound) {
    *p_C_bound = mat_create (m, n);
    mat_setZero (m, n, *p_C_bound);
  }
}

/* ------------------------------------------------------------ */

static
int
checkEnvEnabled__ (const char* var, int def_val)
{
  /* The MPI standard does not mandate that environment variables at
   * job launch will be set for all processes, only the root. Thus,
   * this routine checks it on the root and then uses a broadcast to
   * send the results to all processes.
   */
  int rank = mpih_getRank (MPI_COMM_WORLD);
  int enabled = 0;
  if (rank == 0)
    enabled = env_isEnabled ("VERIFY", 1) || env_getInt ("VERIFY", 1);
  MPI_Bcast (&enabled, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (rank == 0)
    mpih_debugmsg (MPI_COMM_WORLD, "'%s' %senabled...\n",
		   var, enabled ? "" : "not ");
  return enabled;
}

/* ------------------------------------------------------------ */

static
void
summarize__ (int m, int n, int k, int s, const double* t, int n_t,
	     MPI_Comm comm, int debug)
{
  MPI_Barrier (comm);
  FILE* fp = debug ? stderr : stdout;
  int P = mpih_getSize (comm);
  int rank = mpih_getRank (comm);
  if (rank == 0)
    fprintf (fp, "%s%d %d %d %d %d", debug ? "DEBUG: " : "", m, n, k, s, P);
  for (int i = 0; i < n_t; ++i) {
    double* tt = (double *)t; /* remove cast */
    double ti_min;
    MPI_Reduce (&tt[i], &ti_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    double ti_max;
    MPI_Reduce (&tt[i], &ti_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    double ti_sum;
    MPI_Reduce (&tt[i], &ti_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    if (rank == 0)
      fprintf (fp, " %g %g %g", ti_min, ti_max, ti_sum / P);
  }
  if (rank == 0)
    fprintf (fp, "\n");
  MPI_Barrier (comm);
}

/* ------------------------------------------------------------ */
void
benchmark__ (int m, int n, int k, int P_row, int P_col, int s)
{
  if (!checkEnvEnabled__ ("BENCH", 1)) return;

  MPI_Comm comm2d = summa_createTopology (MPI_COMM_WORLD, P_row, P_col);
  int rank = mpih_getRank (comm2d);

  if (rank == 0) mpih_debugmsg (comm2d, "Beginning benchmark...\n");

  /* Create a synthetic problem to benchmark. */
  double* A_local = summa_alloc (m, k, comm2d);
  double* B_local = summa_alloc (k, n, comm2d);
  double* C_local = summa_alloc (m, n, comm2d);

  summa_randomize (m, k, A_local, comm2d);
  summa_randomize (k, n, B_local, comm2d);

  const int TOTAL = 0;
  const int COMP = 1;
  const int COMM = 2;
  double t[3];  bzero (t, sizeof (t));

  const int MAX_TRIALS = 10;
  if (rank == 0)
    mpih_debugmsg (comm2d, "Multiplying [%d trials]...\n", MAX_TRIALS);

  for (int trial = 0; trial < MAX_TRIALS; ++trial) {
    summa_setZero (m, n, C_local, comm2d);
    double t_start = MPI_Wtime ();
    summa_mult (m, n, k, s, A_local, B_local, C_local, comm2d,
		&t[COMP], &t[COMM]);
    t[TOTAL] += MPI_Wtime () - t_start;
  }

  if (rank == 0) mpih_debugmsg (comm2d, "Done!\n");
  summarize__ (m, n, k, s, t, 3, comm2d, 0);

  summa_free (A_local, comm2d);
  summa_free (B_local, comm2d);
  summa_free (C_local, comm2d);
  summa_freeTopology (comm2d);
}

/* eof */
