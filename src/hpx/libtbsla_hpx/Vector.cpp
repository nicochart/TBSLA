#include <tbsla/hpx/Vector.hpp>
#include <hpx/hpx.hpp>

typedef hpx::components::component<tbsla::hpx_::server::Vector> Vector_server_type;
HPX_REGISTER_COMPONENT(Vector_server_type, Vector_server);

typedef tbsla::hpx_::server::Vector::get_data_action get_Vector_data_action;
HPX_REGISTER_ACTION(get_Vector_data_action);

static tbsla::hpx_::client::Vector tbsla::hpx_::detail::reduce_part(tbsla::hpx_::client::Vector const& vc1, tbsla::hpx_::client::Vector const& vc2) {
  using hpx::dataflow;
  using hpx::util::unwrapping;

  hpx::shared_future<tbsla::hpx_::detail::Vector> vd1 = vc1.get_data();
  hpx::shared_future<tbsla::hpx_::detail::Vector> vd2 = vc2.get_data();
  return dataflow(hpx::launch::async,
    unwrapping([vc1](tbsla::hpx_::detail::Vector const& v1, tbsla::hpx_::detail::Vector const& v2) -> tbsla::hpx_::client::Vector {
      std::vector<double> vv1 = v1.get_vect();
      std::vector<double> vv2 = v2.get_vect();
      std::vector<double> rv(vv1.size());
      if(vv1.size() == vv2.size()) {
         for(int i = 0; i < vv1.size(); i++)
           rv[i] = vv1[i] + vv2[i];
      }
      tbsla::hpx_::detail::Vector r(rv);
      return tbsla::hpx_::client::Vector(vc1.get_id(), r);
    }),
    vd1, vd2);
}

HPX_PLAIN_ACTION(tbsla::hpx_::detail::reduce_part, reduce_part_action);

tbsla::hpx_::client::Vector add_vectors(hpx::id_type where, tbsla::hpx_::client::Vector v1, tbsla::hpx_::client::Vector v2) {
  using hpx::util::placeholders::_1;
  using hpx::util::placeholders::_2;
  using hpx::dataflow;

  reduce_part_action reduce_act;
  auto Op = hpx::util::bind(reduce_act, where, _1, _2);
  return dataflow(hpx::launch::async, Op, v1 ,v2);
}

std::vector<tbsla::hpx_::client::Vector> reduce_recursion(std::vector<tbsla::hpx_::client::Vector> vcs) {
  if(vcs.size() < 1) {
    throw "error in recursive reduction";
  } else if(vcs.size() == 1) {
    return vcs;
  } else {
    std::vector<tbsla::hpx_::client::Vector> new_vcs;
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
      auto Op = hpx::util::bind(reduce_act, localities[i * nl / div2], _1, _2);
      new_vcs.push_back(dataflow(hpx::launch::async, Op, vcs[2 * i], vcs[2 * i + 1]));
    }
    if(mod == 1) {
      new_vcs.push_back(vcs.back());
    }
    return reduce_recursion(new_vcs);
  }
}

tbsla::hpx_::client::Vector reduce(std::vector<tbsla::hpx_::client::Vector> vcs) {
  return reduce_recursion(vcs).front();
}
