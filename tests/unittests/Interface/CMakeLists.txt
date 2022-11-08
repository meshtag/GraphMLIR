add_executable(
	InterfaceTests
	GraphContainerTest.cpp
	ContainerTest.cpp
)

target_link_libraries(InterfaceTests gtest gtest_main pthread BFS)

add_dependencies(InterfaceTests Container GraphContainer)

include(GoogleTest)
gtest_discover_tests(InterfaceTests)
