set(Boost_USE_STATIC_LIBS ON)

find_package(Boost COMPONENTS filesystem system unit_test_framework REQUIRED)

add_executable(my_test my_test.cpp ${SOURCES})

target_link_libraries(my_test ${Boost_LIBRARIES})

add_test(MyTest my_test COMMAND my_test)