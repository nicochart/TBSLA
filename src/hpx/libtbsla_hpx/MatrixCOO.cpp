#include <tbsla/hpx/MatrixCOO.hpp>

typedef hpx::components::component<MatrixCOO_server> MatrixCOO_server_type;
HPX_REGISTER_COMPONENT(MatrixCOO_server_type, MatrixCOO_server);

typedef MatrixCOO_server::get_data_action get_MatrixCOO_data_action;
HPX_REGISTER_ACTION(get_MatrixCOO_data_action);


static Vector_client spmv_part(MatrixCOO_client const& A_p, Vector_client const& v_p, Vector_client const& r_p) {
  using hpx::dataflow;
  using hpx::util::unwrapping;

  hpx::shared_future<tbsla::hpx::MatrixCOO> A_data = A_p.get_data();
  hpx::shared_future<Vector_data> v_data = v_p.get_data();
  return dataflow(hpx::launch::async,
    unwrapping([r_p](tbsla::hpx::MatrixCOO const& A, Vector_data const& v) -> Vector_client {
      Vector_data r(A.spmv(v.get_vect(), 0));
      return Vector_client(r_p.get_id(), r);
    }),
    A_data, v_data);
}

HPX_PLAIN_ACTION(spmv_part, spmv_part_action);

Vector_client do_spmv(std::size_t N, std::string matrix_file) {
  using hpx::dataflow;

  std::vector<hpx::id_type> localities = hpx::find_all_localities();
  std::size_t nl = localities.size();    // Number of localities

  std::vector<MatrixCOO_client> tiles;
  tiles.resize(N);

  std::vector<Vector_client> spmv_res;
  spmv_res.resize(N);

  for (std::size_t i = 0; i != N; ++i) {
    tiles[i] = MatrixCOO_client(localities[i % nl], i, N, matrix_file);
  }

  tiles[0].get_data().wait();
  tbsla::hpx::MatrixCOO m = tiles[0].get_data().get();
  int n_row = m.get_n_row();

  for (std::size_t i = 0; i != N; ++i) {
    spmv_res[i] = Vector_client(localities[i % nl], n_row);
  }

  Vector_client v(localities[0], n_row);
  spmv_part_action act_spmv;

  for (std::size_t i = 0; i != N; ++i) {
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    using hpx::util::placeholders::_3;
    auto Op = hpx::util::bind(act_spmv, localities[i % nl], _1, _2, _3);
    spmv_res[i] = dataflow(hpx::launch::async, Op, tiles[i], v, spmv_res[i]);
  }

  return reduce(spmv_res);
}

