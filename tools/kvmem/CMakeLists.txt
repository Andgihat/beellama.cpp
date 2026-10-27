if (NOT LLAMA_KVMEM)
    return()
endif()

add_executable(llama-kvmem-cli
    llama-kvmem-cli.cpp
    kvmem-spec.cpp
)
target_include_directories(llama-kvmem-cli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/adapter
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../kvmem/include
)
target_link_libraries(llama-kvmem-cli PRIVATE llama llama-common kvmem)
target_compile_features(llama-kvmem-cli PRIVATE cxx_std_17)

add_executable(llama-kvmem-server
    kvmem-server-metrics.cpp
    llama-kvmem-server.cpp
    llama-kvmem-driver.cpp
    kvmem-spec.cpp
    kvmem-vision.cpp
    kvmem-responses.cpp
    kvmem-responses-stream.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../server/server-common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../server/server-chat.cpp
)
target_include_directories(llama-kvmem-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../kvmem/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/adapter
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../server
    ${CMAKE_CURRENT_SOURCE_DIR}/../mtmd
    ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor
    ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/cpp-httplib
)
target_link_libraries(llama-kvmem-server PRIVATE llama llama-common cpp-httplib mtmd kvmem ${CMAKE_DL_LIBS})
if (GGML_CUDA)
    find_package(CUDAToolkit QUIET)
    target_include_directories(llama-kvmem-server PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()
target_compile_features(llama-kvmem-server PRIVATE cxx_std_17)
if (WIN32)
    target_compile_definitions(llama-kvmem-server PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
if (MSVC)
    target_compile_options(llama-kvmem-cli PRIVATE /EHsc)
    target_compile_options(llama-kvmem-server PRIVATE /EHsc)
endif()
