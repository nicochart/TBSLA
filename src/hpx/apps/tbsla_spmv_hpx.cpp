#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <tbsla/hpx/MatrixCOO.hpp>

int hpx_main(hpx::program_options::variables_map& vm)
{
    std::uint64_t N = vm["N"].as<std::uint64_t>();
    std::string matrix_input = vm["matrix_input"].as<std::string>();

    std::vector<hpx::id_type> localities = hpx::find_all_localities();
    std::size_t nl = localities.size();    // Number of localities

    if (N < nl)
    {
        std::cout << "The number of tiles should not be smaller than "
                     "the number of localities"
                  << std::endl;
        return hpx::finalize();
    }

    std::uint64_t t = hpx::util::high_resolution_clock::now();

    Vector_client r = do_spmv(N, matrix_input);
    r.get_data().wait();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()(
        "print-input", "print matrix and vector")("N",
        value<std::uint64_t>()->default_value(10),
        "Dimension of the submatrices")("matrix_input",
        value<std::string>()->default_value(""),
        "file containing the matrix (default : empty)");

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
