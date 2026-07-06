cmake_minimum_required ( VERSION 3.5 )
project ( ml_nips CXX )

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CXX_EXTENSIONS OFF)

if ( APPLE )
    set ( CMAKE_MACOSX_RPATH OFF )
endif ( APPLE )

include ( FindPkgConfig )
pkg_search_module ( SNORT3 REQUIRED snort>=3 )

add_library (
    ml_nips MODULE
    ml_nips.cc
)

if ( APPLE )
    set_target_properties (
        ml_nips
        PROPERTIES
            LINK_FLAGS "-undefined dynamic_lookup"
    )
endif ( APPLE )

set_target_properties (
    ml_nips
    PROPERTIES
        PREFIX ""
)

target_include_directories (
    ml_nips PUBLIC
    ${SNORT3_INCLUDE_DIRS}
    /home/nadyahra/tensorflow
    /home/nadyahra/tensorflow/tflite_build/flatbuffers/include
    /home/nadyahra/tensorflow/tflite_build/abseil-cpp
    /opt/onnxruntime/include
)

target_link_libraries (
    ml_nips
    /home/nadyahra/tensorflow/tflite_build/libtensorflow-lite.so
    /opt/onnxruntime/lib/libonnxruntime.so
    pthread
    dl
)

install (
    TARGETS ml_nips
    LIBRARY
        DESTINATION "${INSPECTOR_INSTALL_PATH}"
)
