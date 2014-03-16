#!/bin/sh

CurrentDir=$PWD

if ["$1" = ""]; then
  echo "using default generator"
  GeneratorOptions=""
else
  echo "using $1"
  GeneratorOptions=-G"$1"
fi

CMAKE_OGG_INC_DIR=${CurrentDir}/Deps/include
CMAKE_VORBIS_INC_DIR=${CMAKE_OGG_INC_DIR}
CMAKE_OPENAL_INC_DIR=${CurrentDir}/../../ThirdParty-mod/openal-soft/include/AL

CMAKE_OGG_LIB_DIR=${CurrentDir}/Deps/lib
CMAKE_VORBIS_LIB_DIR=${CMAKE_OGG_LIB_DIR}
CMAKE_OPENAL_LIB_DIR=${CMAKE_OGG_LIB_DIR}

DepsIncPathOptions="-DCMAKE_OGG_INC_DIR=${CMAKE_OGG_INC_DIR} -DCMAKE_VORBIS_INC_DIR=${CMAKE_VORBIS_INC_DIR} -DCMAKE_OPENAL_INC_DIR=${CMAKE_OPENAL_INC_DIR}"
DepsLibPathOptions="-DCMAKE_OGG_LIB_DIR=${CMAKE_OGG_LIB_DIR} -DCMAKE_VORBIS_LIB_DIR=${CMAKE_VORBIS_LIB_DIR} -DCMAKE_OPENAL_LIB_DIR=${CMAKE_OPENAL_LIB_DIR}"


mkdir Debug
cd Debug
echo cmake -DCMAKE_BUILD_TYPE=Debug "${GeneratorOptions}" ${DepsIncPathOptions} ${DepsLibPathOptions} ..
cmake -DCMAKE_BUILD_TYPE=Debug "${GeneratorOptions}" ${DepsIncPathOptions} ${DepsLibPathOptions} ..

cd $CurrentDir

echo "go to $CurrentDir/Debug to build the project"
