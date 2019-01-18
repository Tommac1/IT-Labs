cmake_minimum_required (VERSION 3.7)
project (main)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)

include_directories (main "/home/anon/Repositories/json/include/nlohmann"
    "/home/anon/Repositories/json/include"
    ${PROJECT_SOURCE_DIR}/include)

file (GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
file (GLOB_RECURSE HEADERS ${PROJECT_SOURCE_DIR}/include/*.h)

add_executable (main ${SRC_FILES} ${HEADERS})
target_link_libraries (main pthread)

