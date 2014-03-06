#!/bin/bash
cd ../HQEngine
source setEnv.sh
$ndk_build_script -j 8 "$@"

cd ../HQAudio
$ndk_build_script -j 8 "$@"

cd ../HQSceneManagement 
$ndk_build_script -j 8 "$@"

cd ../test
$ndk_build_script -j 8 "$@"