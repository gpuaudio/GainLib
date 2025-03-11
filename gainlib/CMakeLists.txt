# Component name
set(component_name gainlib)


include(FetchContent)
cmake_policy(SET CMP0135 NEW)

# ENGINE API
FetchContent_Declare(
    engine_api
    URL https://github.com/gpuaudio/platform_headers/releases/download/v0.0.1/engine_api-v2.2.0_63.zip
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/engine_api
)
FetchContent_MakeAvailable(engine_api)

# GPU AUDIO CLIENT
FetchContent_Declare(
    gpu_audio_client
    URL https://github.com/gpuaudio/platform_headers/releases/download/v0.0.1/gpu_audio_client-v0.1.1_12.zip
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu_audio_client
)
FetchContent_MakeAvailable(gpu_audio_client)

# JSON
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Google Tests
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# Component project
add_library(${component_name} STATIC)

set(target_src
    src/GPUCreate.cpp
    src/GPUGainProcessor.cpp
)

set(target_headers
    src/GPUGainProcessor.h
)

# Source files
target_sources(${component_name} PRIVATE
    ${target_src}
    ${target_headers}
)

# Include directories
target_include_directories(${component_name} PRIVATE
    ../include
    include
)

# Header-only dependencies if they were fetched from the top-level project
if (DEFINED FETCH_LOCATION)
    target_include_directories(${component_name} PUBLIC
        ${FETCH_LOCATION}/include
    )
endif ()

# Compile definitions
if (WIN32)
    set(win_common_private_compile_definitions
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
endif ()

target_compile_definitions(${component_name} PRIVATE
    ${win_common_private_compile_definitions}
)

# Link libraries
target_link_libraries(${component_name} PRIVATE nlohmann_json::nlohmann_json)

# Unit tests
set(tests_name ${component_name}_tests)

add_executable(${tests_name})

# Source files
target_sources(${tests_name} PRIVATE
    tests/TestCommon.h
    tests/GPUGainProcessorTests.cpp
)

# Include directories
target_include_directories(${tests_name} PRIVATE
    ../include
    include
)

# Compile definitions
if (WIN32)
    set(win_common_private_compile_definitions
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
elseif(APPLE)
    set(apple_common_private_compile_definitions
        GPU_AUDIO_MAC
    )
endif ()

target_compile_definitions(${tests_name} PRIVATE
    ${win_common_private_compile_definitions}
    ${apple_common_private_compile_definitions}
    BUILD_TYPE="$<CONFIG>"
)

# Link libraries
target_link_libraries(${tests_name} PRIVATE
    ${component_name}
    gtest_main
    nlohmann_json::nlohmann_json
)

gtest_add_tests(TARGET ${tests_name})

set_property(TARGET ${component_name} PROPERTY COMPILE_WARNING_AS_ERROR OFF)
set_property(TARGET ${tests_name} PROPERTY COMPILE_WARNING_AS_ERROR OFF)
