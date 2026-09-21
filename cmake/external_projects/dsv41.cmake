# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Keep GGML's options and compiler settings out of other vLLM targets.
enable_language(C)
function(vllm_add_dsv41_backends)
  include(FetchContent)
  find_package(Git REQUIRED)
  find_package(OpenMP REQUIRED COMPONENTS C CXX)
  find_package(Threads REQUIRED)
  find_package(CUDAToolkit REQUIRED)

  set(wrapper_dir "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../csrc/cpu/dsv41")
  FetchContent_Declare(dsv41_ik
    URL "https://codeload.github.com/ikawrakow/ik_llama.cpp/tar.gz/3bb386eb68ffee0a5dc7db21da0735d594929eeb"
    URL_HASH SHA256=ad4bad0bbfd6e664867fca53c95e2d2b285d440d33cfaa938b1d93c4d1d6313c
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    # Populate only: we need GGML, not the upstream executables or CUDA backend.
    SOURCE_SUBDIR vllm-unused)
  FetchContent_MakeAvailable(dsv41_ik)

  set(patch "${wrapper_dir}/ik-pre-silu-clamp.patch")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${patch}")
  execute_process(COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${patch}"
    WORKING_DIRECTORY "${dsv41_ik_SOURCE_DIR}"
    RESULT_VARIABLE patched OUTPUT_QUIET ERROR_QUIET)
  if(NOT patched EQUAL 0)
    execute_process(COMMAND "${GIT_EXECUTABLE}" apply "${patch}"
      WORKING_DIRECTORY "${dsv41_ik_SOURCE_DIR}"
      COMMAND_ERROR_IS_FATAL ANY)
  endif()

  set(BUILD_SHARED_LIBS OFF)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(GGML_CCACHE OFF)
  set(GGML_CUDA OFF)
  set(GGML_NATIVE OFF)
  set(GGML_AVX ON)
  set(GGML_AVX2 ON)
  set(GGML_FMA ON)
  set(GGML_F16C ON)
  set(GGML_AVXVNNI OFF)
  set(GGML_AVX512 OFF)
  set(GGML_AVX512_VBMI OFF)
  set(GGML_AVX512_VNNI OFF)
  set(GGML_AVX512_BF16 OFF)
  set(GGML_OPENMP ON)
  set(GGML_IQK_MUL_MAT ON)
  set(GGML_IQK_FLASH_ATTENTION ON)
  set(GGML_IQK_FA_ALL_QUANTS OFF)
  set(GGML_BUILD_TESTS OFF)
  set(GGML_BUILD_EXAMPLES OFF)
  add_subdirectory("${dsv41_ik_SOURCE_DIR}/ggml"
    "${CMAKE_CURRENT_BINARY_DIR}/dsv41-ggml" EXCLUDE_FROM_ALL)

  add_library(libdsv41_ik SHARED "${wrapper_dir}/ggml_moe.cpp")
  target_compile_definitions(libdsv41_ik PRIVATE DSV41_IK)
  target_include_directories(libdsv41_ik PRIVATE "${dsv41_ik_SOURCE_DIR}/ggml/src")
  target_compile_options(libdsv41_ik PRIVATE -mavx2 -mfma -mf16c)
  target_link_libraries(libdsv41_ik PRIVATE ggml OpenMP::OpenMP_CXX
    Threads::Threads ${CMAKE_DL_LIBS})
  target_link_options(libdsv41_ik PRIVATE -Wl,--exclude-libs,ALL -Wl,-Bsymbolic)

  # setup.py defaults to RelWithDebInfo; retain the tested CPU optimization level.
  foreach(target ggml libdsv41_ik)
    target_compile_options(${target} PRIVATE
      $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-O3>)
  endforeach()

  add_library(libdsv41_cuda SHARED "${wrapper_dir}/cuda_host_moe.cpp"
    "${wrapper_dir}/engram_ssd.cpp")
  target_link_libraries(libdsv41_cuda PRIVATE CUDA::cudart)
  set_target_properties(libdsv41_cuda PROPERTIES
    INSTALL_RPATH "$ORIGIN/../nvidia/cuda_runtime/lib;$ORIGIN/../nvidia/cu13/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE)

  foreach(target libdsv41_ik libdsv41_cuda)
    set_target_properties(${target} PROPERTIES PREFIX "")
    install(TARGETS ${target} LIBRARY DESTINATION vllm COMPONENT ${target})
  endforeach()
  install(FILES "${dsv41_ik_SOURCE_DIR}/LICENSE"
    DESTINATION vllm/third_party/ik_llama COMPONENT libdsv41_ik)
  message(STATUS "DeepSeek hybrid: pinned IK, AVX2/FMA/F16C, OpenMP, CUDA host callbacks")
endfunction()

vllm_add_dsv41_backends()
