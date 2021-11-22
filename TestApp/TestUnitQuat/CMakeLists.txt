cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(TestUnitQuat)

set(SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/transformation_estimation_uq.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/transformation_estimation_uq.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/pose_est_Lu.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/pose_est_Lu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

add_executable(TestUnitQuat ${SRC_FILES})

find_package(PCL 1.8 REQUIRED)

if( PCL_FOUND )
  # [C/C++]>[General]>[Additional Include Directories]
  include_directories( 
    ${PCL_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src 
  )

  # [C/C++]>[Preprocessor]>[Preprocessor Definitions]
  add_definitions( ${PCL_DEFINITIONS} )
  
  # For Use Not PreCompiled Features 
  #add_definitions( -DPCL_NO_PRECOMPILE )

  # [Linker]>[General]>[Additional Library Directories]
  link_directories( ${PCL_LIBRARY_DIRS} )

  # [Linker]>[Input]>[Additional Dependencies]
  target_link_libraries(TestUnitQuat ${PCL_LIBRARIES})  



endif()