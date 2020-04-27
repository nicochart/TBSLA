find_package(MPI REQUIRED)
find_package(PARSEC REQUIRED)

include(PaRSECConfig)
include(PaRSECTargets)
include(ParsecCompilePTG)
include(ParsecConfigVersion)

add_subdirectory(tests)
add_subdirectory(libtbsla_parsec)
