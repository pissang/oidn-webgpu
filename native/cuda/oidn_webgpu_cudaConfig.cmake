include(CMakeFindDependencyMacro)
find_dependency(oidn_webgpu CONFIG)
find_dependency(CUDAToolkit 12.5)
include("${CMAKE_CURRENT_LIST_DIR}/oidn_webgpu_cudaTargets.cmake")
