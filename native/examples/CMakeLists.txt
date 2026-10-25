cmake_minimum_required(VERSION 3.20)
project(oidn_webgpu_c_examples LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
# Shipped Windows libraries use the DLL CRT, also for a Debug host build.
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
if(APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path/../lib")
elseif(UNIX)
  set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
endif()

find_package(oidn_webgpu CONFIG REQUIRED)

add_executable(denoise denoise.c)
target_link_libraries(denoise PRIVATE oidn_webgpu::oidn_webgpu)
add_executable(plan plan.c)
target_link_libraries(plan PRIVATE oidn_webgpu::oidn_webgpu_core)
set(examples denoise plan)
set(shared_examples denoise)
if(APPLE)
  enable_language(OBJC)
  add_executable(metal_gpu metal_gpu.m)
  # Older supported CMake releases omit Objective-C runtime RPATH flags.
  # The C driver links these ObjC objects too and preserves build/install RPATH.
  set_target_properties(metal_gpu PROPERTIES LINKER_LANGUAGE C)
  target_compile_options(metal_gpu PRIVATE -fobjc-arc)
  target_link_libraries(metal_gpu PRIVATE oidn_webgpu::oidn_webgpu "-framework Metal" "-framework Foundation")
  list(APPEND examples metal_gpu)
  list(APPEND shared_examples metal_gpu)
endif()
if(WIN32)
  add_executable(validate_windows validate_windows.c)
  target_link_libraries(validate_windows PRIVATE oidn_webgpu::oidn_webgpu)
  list(APPEND examples validate_windows)
  list(APPEND shared_examples validate_windows)
  foreach(example IN LISTS shared_examples)
    add_custom_command(TARGET ${example} POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "$<TARGET_FILE:oidn_webgpu::oidn_webgpu>" "$<TARGET_FILE_DIR:${example}>"
      VERBATIM)
  endforeach()
  set(runtime_destination bin)
else()
  set(runtime_destination lib)
endif()

install(TARGETS ${examples} RUNTIME DESTINATION bin)
install(FILES "$<TARGET_FILE:oidn_webgpu::oidn_webgpu>" DESTINATION "${runtime_destination}")
