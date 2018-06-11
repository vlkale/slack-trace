cmake_minimum_required(VERSION 2.8)

set(CMAKE_C_COMPILER /usr/bin/gcc)

set(WRAP ${PROJECT_SOURCE_DIR}/wrap/wrap.py)
include(wrap/WrapConfig.cmake)

find_package(adept_utils REQUIRED)
find_package(MPI REQUIRED)

add_wrapped_file(slack-trace.C slack-trace.w)
add_wrapped_file(slack-trace.h slack-trace.h.w)

# target_link_libraries(slackconscious ssl unwind)

include(CheckFunctionExists)
# find_package(OpenMP)

#set(DUSCHED_SRCS
#  slack-trace.C ${CMAKE_CURRENT_BINARY_DIR}/slack-trace.h)

#add_library(d-usched-mpi-only ${DUSCHED_SRCS})
#target_link_libraries(d-usched-mpi-only ${MPI_LIBRARIES} adept_timing)

set(ROSE_HOME /usr/global/tools/rose-compiler/$ENV{SYS_TYPE}/rose-0.9.5a-20390)
set(ROSE_LIBRARIES ${ROSE_HOME}/lib/librose.so)
set(ROSE_INCLUDE_PATH ${ROSE_HOME}/include)

set(SHASSL_LIBRARIES /usr/lib64/libssl.so.10)
set(UNWIND_LIBRARIES /usr/lib64/libunwind.so.8)

add_library(slackconscious SHARED
  slack-trace.h
  slack-trace.C)

# set(ADEPTUTILS_LIBRARIES ~/adept-utils/${SYS_TYPE}/timing/libadept_timing.so)
target_link_libraries(
  slackconscious
  ${SHASSL_LIBRARIES}
  ${UNWIND_LIBRARIES}
  ${MPI_C_LIBRARIES}
  adept_timing)

function(add_mpi_executable exe_name src_name)
  add_executable(${exe_name} ${src_name})
  target_link_libraries(${exe_name} ${MPI_C_LIBRARIES})
endfunction()

function(add_rose_translator exe_name src_name)
  add_executable(${exe_name} ${src_name})
  target_link_libraries(${exe_name} ${ROSE_LIBRARIES} ${MPI_C_LIBRARIES} slackconscious)
endfunction()

#add_mpi_executable(dotProduct dotProduct.C)
#add_mpi_executable(dotProd dotProd.c)
#add_rose_translator(identityTranslator identityTranslator.C)

#add_rose_translator(myTranslator myTranslator.C)

#install rules

add_rose_translator(uSchedTrans uSchedTrans.C)

install(TARGETS uSchedTrans    DESTINATION bin)
install(TARGETS slackconscious DESTINATION lib)

include_directories(
  ${PROJECT_BINARY_DIR}
  ${MPI_INCLUDE_PATH}
  ${MPI_C_INCLUDE_PATH}
  ${ROSE_INCLUDE_PATH}
  ${adept_utils_INCLUDE_DIR}
  ${SLACKCONSCIOUS_INCLUDE_PATH}
)

