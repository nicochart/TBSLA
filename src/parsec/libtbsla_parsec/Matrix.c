#include <tbsla/parsec/Matrix.h>
#include <string.h>

struct parsecMatrix *newParsecMatrixCDIAG(MPI_Comm comm, int matrix_format, int nr, int nc, int c, int gr, int gc) {
  struct parsecMatrix * pm = malloc(sizeof(struct parsecMatrix));

  int world = 1, rank = 0;
  MPI_Comm_size(comm, &world);
  MPI_Comm_rank(comm, &rank);

  pm->matrices = parsec_hash_datadist_create(world, rank);
  pm->size = 0;
  pm->max_size = 0;
  pm->matrix_format = matrix_format;

  void* data;
  size_t data_size;
  for (int i = 0; i < gr; i++) {
    for (int j = 0; j < gc; j++) {
      data = NULL;
      data_size = 0;
      if (i * gc + j % world == rank) {
        switch(matrix_format)
        {
          case TBSLA_PaRSEC_DENSE:
            break;
          case TBSLA_PaRSEC_COO:
            break;
          case TBSLA_PaRSEC_CSR:
            break;
          case TBSLA_PaRSEC_ELL:
            break;
        }
        data_size = 1000;
        data = malloc(data_size);
        pm->size += data_size;
        if (data_size > pm->max_size) pm->max_size = data_size;
      }
      parsec_hash_datadist_set_data(pm->matrices, data, i * gc + j, 0, i * gc + j % world, data_size);
    }
  }

  return pm;
}
