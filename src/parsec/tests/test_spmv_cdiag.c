#include "parsec.h"
#include "parsec/runtime.h"
#include "parsec/data_dist/hash_datadist.h"

#include "spmv.h"
#include <tbsla/parsec/Matrix.h>
#include <tbsla/parsec/Vector.h>

int main(int argc, char **argv) {
  parsec_context_t* parsec;
  parsec_taskpool_t* op;
  parsec_spmv_taskpool_t *tp;

  int cores = -1, world = 1, rank = 0, rc;
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int nr = 100, nc = 100, gr = 5, gc = 5, data_size = 0;

  parsec = parsec_init(cores, &argc, &argv);

  struct parsecMatrix * pm = newParsecMatrixCDIAG(MPI_COMM_WORLD, TBSLA_PaRSEC_COO, nr, nc, 300, gr, gc);
  struct parsecVector * pv = newParsecVector(MPI_COMM_WORLD, nr, gr, gc);

  tp = parsec_spmv_new(pm->matrices, pv->vectors, gr, gc);

  parsec_arena_construct(tp->arenas[PARSEC_spmv_VECTOR_ARENA],
                         pv->max_size, PARSEC_ARENA_ALIGNMENT_SSE,
                         MPI_BYTE);

  parsec_arena_construct(tp->arenas[PARSEC_spmv_MATRIX_ARENA],
                         pm->max_size, PARSEC_ARENA_ALIGNMENT_SSE,
                         MPI_BYTE);

  rc = parsec_context_add_taskpool(parsec, (parsec_taskpool_t *) tp);
  PARSEC_CHECK_ERROR(rc, "parsec_context_add_taskpool");

  rc = parsec_context_start(parsec);
  PARSEC_CHECK_ERROR(rc, "parsec_context_start");
  rc = parsec_context_wait(parsec);
  PARSEC_CHECK_ERROR(rc, "parsec_context_wait");
  parsec_taskpool_free((parsec_taskpool_t *) tp);
  parsec_fini(&parsec);
  MPI_Finalize();
  printf("end \n");

  return 0;


}
