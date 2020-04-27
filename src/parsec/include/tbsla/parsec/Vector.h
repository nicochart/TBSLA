#ifndef TBSLA_PaRSEC_Vector
#define TBSLA_PaRSEC_Vector

#include "parsec/data_dist/hash_datadist.h"

struct parsecVector {
  size_t size;
  size_t max_size;
  parsec_hash_datadist_t *vectors;
};

struct parsecVector *newParsecVector(MPI_Comm comm, int nr, int gr, int gc);

#endif
