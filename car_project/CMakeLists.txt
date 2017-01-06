#CMakeLists.txt to build project using cmake
#run $cmake CMakeLists.txt

cmake_minimum_required(VERSION 2.6)



project(pgm)

find_package( OpenCV REQUIRED )


include_directories(${CMAKE_CURRENT_BINARY_DIR})


file(
GLOB_RECURSE
source_files
src/*.[cht]pp
)

SET(CMAKE_BUILD_TYPE Debug)
ADD_DEFINITIONS( -O2 -Wall -Wextra -std=c++11)



add_executable(
  pgm
  ${source_files}
)


TARGET_LINK_LIBRARIES(pgm -lm -ldl -lGLEW ${OpenCV_LIBS})
