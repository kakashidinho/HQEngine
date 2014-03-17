
CurrentDir=$PWD

if [ -z "$1" ]; then
  echo "using default generator"
  GeneratorOption=""
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

mkdir Release
cd Release
cmake -DCMAKE_BUILD_TYPE=Release  "${GeneratorOption}" ${DepsIncPathOptions} ${DepsLibPathOptions} ..

cd $CurrentDir

echo "go to $CurrentDir/Release to build the project"
