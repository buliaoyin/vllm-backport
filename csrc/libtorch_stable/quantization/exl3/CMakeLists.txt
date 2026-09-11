cmake_minimum_required(VERSION 3.26)

get_filename_component(EXL3_VLLM_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_LIST_DIR)
  project(vllm_exl3 LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CUDA_STANDARD 20)
  include("${EXL3_VLLM_ROOT}/cmake/utils.cmake")
  if(NOT VLLM_PYTHON_EXECUTABLE)
    message(FATAL_ERROR "Set VLLM_PYTHON_EXECUTABLE to the vLLM environment's Python")
  endif()
  find_python_from_executable("${VLLM_PYTHON_EXECUTABLE}" "3.10;3.11;3.12;3.13;3.14")
  append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")
  find_package(Torch REQUIRED)
  clear_cuda_gencode_flags(EXL3_UNUSED_ARCH_FLAGS)
endif()

set(EXL3_CUDA_ARCHS 80)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  list(APPEND EXL3_CUDA_ARCHS 120)
endif()

define_extension_target(
  _exl3_C
  DESTINATION vllm
  LANGUAGE CUDA
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/moe.cu" "${CMAKE_CURRENT_LIST_DIR}/decode.cu"
  COMPILE_FLAGS --use_fast_math --expt-relaxed-constexpr --expt-extended-lambda
  ARCHITECTURES ${EXL3_CUDA_ARCHS}
  INCLUDE_DIRECTORIES "${EXL3_VLLM_ROOT}/csrc"
  USE_SABI 3
  WITH_SOABI)
target_compile_definitions(_exl3_C PRIVATE
  USE_CUDA TORCH_TARGET_VERSION=0x020B000000000000ULL)
