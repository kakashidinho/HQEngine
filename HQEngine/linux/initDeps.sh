#!/bin/sh

CurrentDir=$PWD
DepsDir=$CurrentDir/Deps

#-------------build libogg-------------------
cd ../../ThirdParty-mod/libogg
chmod +x configure
./configure --prefix=$DepsDir --enable-shared=no CFLAGS=-fPIC
make
make install

#-------------build libvorbis-------------------
cd ../libvorbis
chmod +x configure
./configure --prefix=$DepsDir --with-ogg=$DepsDir --enable-shared=no CFLAGS=-fPIC
make
make install

#-------------build openal-soft-----------------
cd ../openal-soft
mkdir build
cd build
cmake -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$DepsDir/lib ..
make

cd $CurrentDir