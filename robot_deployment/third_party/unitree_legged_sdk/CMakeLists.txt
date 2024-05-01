cmake_minimum_required(VERSION 2.8.3)
project(unitree_legged_sdk)


message("-- CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86_64.*")
  set(ARCH amd64)
endif()
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "aarch64.*")
  set(ARCH arm64)
endif()


include_directories(include)

link_directories(lib)

add_compile_options(-std=c++11)

if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86_64.*")
   set(EXTRA_LIBS -pthread libunitree_legged_sdk_amd64.so lcm)
endif()
if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "aarch64.*")
	set(EXTRA_LIBS -pthread libunitree_legged_sdk_arm64.so lcm)
endif()

set(CMAKE_CXX_FLAGS "-O3 -fPIC")

find_package(catkin QUIET)
if(${catkin_FOUND})
  catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${PROJECT_SOURCE_DIR}/lib/libunitree_legged_sdk_arm64.so  )
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

# one pc one process
add_executable(example_position examples/example_position.cpp)
target_link_libraries(example_position ${EXTRA_LIBS})

add_executable(example_velocity examples/example_velocity.cpp)
target_link_libraries(example_velocity ${EXTRA_LIBS})

add_executable(example_torque examples/example_torque.cpp)
target_link_libraries(example_torque ${EXTRA_LIBS})

add_executable(example_walk examples/example_walk.cpp)
target_link_libraries(example_walk ${EXTRA_LIBS})

# one pc multi process
# add_executable(lcm_server_high examples/lcm_server_high.cpp)
# target_link_libraries(lcm_server_high ${EXTRA_LIBS})

add_executable(lcm_server_low examples/lcm_server_low.cpp)
target_link_libraries(lcm_server_low ${EXTRA_LIBS})

# multi pc
# add_executable(udp_send_test examples/multi_pc_udp_send.cpp)
# target_link_libraries(udp_send_test ${EXTRA_LIBS})

# add_executable(udp_recv_test examples/multi_pc_udp_recv.cpp)
# target_link_libraries(udp_recv_test ${EXTRA_LIBS})

add_executable(lcm_receiver_low examples/lcm_receiver_low.cpp)
target_link_libraries(lcm_receiver_low ${EXTRA_LIBS})
