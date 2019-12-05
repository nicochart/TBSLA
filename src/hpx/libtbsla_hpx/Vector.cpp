#include <tbsla/hpx/Vector.hpp>
#include <hpx/hpx.hpp>

typedef hpx::components::component<Vector_server> Vector_server_type;
HPX_REGISTER_COMPONENT(Vector_server_type, Vector_server);

typedef Vector_server::get_Vector_data_action get_Vector_data_action;
HPX_REGISTER_ACTION(get_Vector_data_action);

static Vector_client reduce_part(Vector_client const& vc1, Vector_client const& vc2) {
  using hpx::dataflow;
  using hpx::util::unwrapping;

  hpx::shared_future<Vector_data> vd1 = vc1.get_data();
  hpx::shared_future<Vector_data> vd2 = vc2.get_data();
  return dataflow(hpx::launch::async,
    unwrapping([vc1](Vector_data const& v1, Vector_data const& v2) -> Vector_client {
      std::vector<double> vv1 = v1.get_vect();
      std::vector<double> vv2 = v2.get_vect();
      std::vector<double> rv(vv1.size());
      if(vv1.size() == vv2.size()) {
         for(int i = 0; i < vv1.size(); i++)
           rv[i] = vv1[i] + vv2[i];
      }
      Vector_data r(rv);
      return Vector_client(vc1.get_id(), r);
    }),
    vd1, vd2);
}

HPX_PLAIN_ACTION(reduce_part, reduce_part_action);

Vector_client add_vectors(hpx::id_type where, Vector_client v1, Vector_client v2) {
  using hpx::util::placeholders::_1;
  using hpx::util::placeholders::_2;
  using hpx::dataflow;

  reduce_part_action reduce_act;
  auto Op = hpx::util::bind(reduce_act, where, _1, _2);
  return dataflow(hpx::launch::async, Op, v1 ,v2);
}

std::vector<Vector_client> reduce_recursion(std::vector<Vector_client> vcs) {
  if(vcs.size() < 1) {
    throw "error in recursive reduction";
  } else if(vcs.size() == 1) {
    return vcs;
  } else {
    std::vector<Vector_client> new_vcs;
    int mod = vcs.size() % 2;
    size_t div2 = vcs.size() / 2;
    new_vcs.reserve(mod + div2);

    reduce_part_action reduce_act;
    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    std::size_t nl = localities.size();

    using hpx::dataflow;
    for(int i = 0; i < div2; i++) {
      using hpx::util::placeholders::_1;
      using hpx::util::placeholders::_2;
      auto Op = hpx::util::bind(reduce_act, localities[i % nl], _1, _2);
      new_vcs.push_back(dataflow(hpx::launch::async, Op, vcs[2 * i], vcs[2 * i + 1]));
    }
    if(mod == 1) {
      new_vcs.push_back(vcs.back());
    }
    return reduce_recursion(new_vcs);
  }
}

Vector_client reduce(std::vector<Vector_client> vcs) {
  return reduce_recursion(vcs).front();
}
