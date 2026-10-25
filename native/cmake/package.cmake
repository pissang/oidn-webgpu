cmake_minimum_required(VERSION 3.20)
include(CMakePackageConfigHelpers)

foreach(variable OIDNW_PACKAGE_DIR OIDNW_VERSION OIDNW_PLATFORM)
  if(NOT DEFINED ${variable})
    message(FATAL_ERROR "${variable} is required")
  endif()
endforeach()

# Config helpers forward arguments through macros, where native Windows
# backslashes would be parsed again as escapes (for example D:\a\...).
string(REPLACE "\\" "/" OIDNW_PACKAGE_DIR "${OIDNW_PACKAGE_DIR}")

# All shipped targets are 64-bit. ConfigVersion also checks the consumer's
# pointer size; the package's target triple selects its CPU architecture.
set(CMAKE_SIZEOF_VOID_P 8)
set(OIDNW_INCLUDE_DIR include)
set(OIDNW_LIBRARY_DIR lib)
set(config_dir "${OIDNW_PACKAGE_DIR}/lib/cmake/oidn_webgpu")
configure_package_config_file(
  "${CMAKE_CURRENT_LIST_DIR}/oidn_webgpuConfig.cmake.in"
  "${config_dir}/oidn_webgpuConfig.cmake"
  INSTALL_PREFIX "${OIDNW_PACKAGE_DIR}"
  INSTALL_DESTINATION lib/cmake/oidn_webgpu
  PATH_VARS OIDNW_INCLUDE_DIR OIDNW_LIBRARY_DIR
)
write_basic_package_version_file(
  "${config_dir}/oidn_webgpuConfigVersion.cmake"
  VERSION "${OIDNW_VERSION}"
  COMPATIBILITY SameMinorVersion
)
