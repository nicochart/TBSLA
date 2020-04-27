#include <tbsla/parsec/Vector.h>

struct parsecVector *newParsecVector(MPI_Comm comm, int nr, int gr, int gc) {
  struct parsecVector * pv = malloc(sizeof(struct parsecVector));

  int world = 1, rank = 0;
  MPI_Comm_size(comm, &world);
  MPI_Comm_rank(comm, &rank);

  pv->vectors = parsec_hash_datadist_create(world, rank);
  pv->size = 0;
  pv->max_size = 0;

  void* data;
  size_t data_size;
  for (int j = 0; j < gc; j++) {
    data = NULL;
    data_size = 0;
    if (j % world == rank) {
      data_size = 1000;
      data = malloc(data_size);
      pv->size += data_size;
      if (data_size > pv->max_size) pv->max_size = data_size;
    }
    parsec_hash_datadist_set_data(pv->vectors, data, j, 0, j % world, data_size);
  }

  return pv;
}
