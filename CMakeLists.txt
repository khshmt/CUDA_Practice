cmake_minimum_required (VERSION 3.8)
project(CUDA_Practice LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_FLAGS "-arch=native")


find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


macro(subdirlist result curdir)
  file(GLOB children RELATIVE ${curdir} ${curdir}/*)
  set(dirlist "")
  foreach(child ${children})
    if(EXISTS ${curdir}/${child}/CMakeLists.txt)
      list(APPEND dirlist ${child})
    endif()
  endforeach()
  set(${result} ${dirlist})
endmacro()

subdirlist(SUBDIRS ${PROJECT_SOURCE_DIR})
foreach(subdir ${SUBDIRS})
    add_subdirectory(${subdir})
endforeach()
