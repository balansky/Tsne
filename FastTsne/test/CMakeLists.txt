set(Boost_USE_STATIC_LIBS ON)

find_package(Boost COMPONENTS filesystem system unit_test_framework REQUIRED)


include_directories(${PROJECT_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/bh_tsne)

add_executable(my_test my_test.cpp ${SOURCES})

target_link_libraries(my_test ${Boost_LIBRARIES} tsne_multicore)

add_test(MyTest my_test COMMAND my_test)