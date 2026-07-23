cmake_minimum_required(VERSION 3.5)

project(vvcam)


file(GLOB SOURCES "src/*.c")
add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC include)