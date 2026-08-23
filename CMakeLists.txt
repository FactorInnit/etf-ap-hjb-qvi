cmake_minimum_required(VERSION 3.16)
project(etf_ap_hjb_qvi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(etf_ap_solve cpp/src/main.cpp)
target_include_directories(etf_ap_solve PRIVATE cpp/include)

if (MSVC)
  target_compile_options(etf_ap_solve PRIVATE /O2 /W4)
else()
  target_compile_options(etf_ap_solve PRIVATE -O3 -Wall -Wextra)
endif()
