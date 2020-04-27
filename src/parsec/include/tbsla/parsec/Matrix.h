#ifndef TBSLA_PaRSEC_Matrix
#define TBSLA_PaRSEC_Matrix

#define TBSLA_PaRSEC_DENSE 1
#define TBSLA_PaRSEC_COO 2
#define TBSLA_PaRSEC_CSR 3
#define TBSLA_PaRSEC_ELL 4

#include "parsec/data_dist/hash_datadist.h"

struct parsecMatrix {
  int matrix_format;
  size_t size;
  size_t max_size;
  parsec_hash_datadist_t *matrices;
};

struct parsecMatrix *newParsecMatrixCDIAG(MPI_Comm comm, int matrix_format, int nr, int nc, int c, int gr, int gc);

#endif
