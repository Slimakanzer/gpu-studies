find_package(Boost 1.40 COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

add_executable(fp32_pooling fp32_pooling.cpp)
target_link_libraries(fp32_pooling ${Boost_LIBRARIES})
target_link_libraries(fp32_pooling examples_common)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/fp32_pooling DESTINATION bin COMPONENT examples)