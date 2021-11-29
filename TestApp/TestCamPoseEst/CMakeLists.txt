cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(TestCamPoseEst)

set(SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/pose_est.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/pose_est.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/pose_est_Lu.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/pose_est_Lu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../inc/pose_est_Ansar.h
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src/pose_est_Ansar.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

add_executable(TestCamPoseEst ${SRC_FILES})

find_package(PCL 1.8 REQUIRED)

if( PCL_FOUND )
  # [C/C++]>[General]>[Additional Include Directories]
  include_directories( 
    ${PCL_INCLUDE_DIRS} 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../inc 
  )

  # [C/C++]>[Preprocessor]>[Preprocessor Definitions]
  add_definitions( ${PCL_DEFINITIONS} )
  
  # For Use Not PreCompiled Features 
  #add_definitions( -DPCL_NO_PRECOMPILE )

  # [Linker]>[General]>[Additional Library Directories]
  link_directories( ${PCL_LIBRARY_DIRS} )

  # [Linker]>[Input]>[Additional Dependencies]
  target_link_libraries(TestCamPoseEst ${PCL_LIBRARIES})  

endif()