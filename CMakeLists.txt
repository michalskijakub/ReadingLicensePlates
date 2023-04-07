cmake_minimum_required(VERSION 3.3.1)

project(HelloOpenCV)

set(OpenCV_DIR "C:/opencv/build")

find_package(OpenCV 4.5.4 REQUIRED)

include_directories(
	${OpenCV_INCLUDE_DIRS}
)

add_executable(HelloOpenCV
	ConsoleApplication2.cpp
	plate_utils.h
	palte_utils.cpp
	
	# You can put here your .h and .cpp files to expand the example.
)

target_link_libraries(HelloOpenCV
	${OpenCV_LIBS}
)