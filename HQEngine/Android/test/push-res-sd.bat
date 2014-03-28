@echo off
set DATA_PATH=/mnt/sdcard/Android/data/hqengineTest.test/files/test
pushd %~dp0..\..\Source\test
adb push setting/Setting.txt %DATA_PATH%/setting/Setting.txt
adb push audio/battletoads-double-dragons-2.ogg %DATA_PATH%/audio/battletoads-double-dragons-2.ogg
adb push audio/crysis_warhead_menu.ogg %DATA_PATH%/audio/crysis_warhead_menu.ogg
adb push image/Marine.dds %DATA_PATH%/image/Marine.dds
adb push image/MarineFlip.pvr %DATA_PATH%/image/MarineFlip.pvr
adb push image/pen2.jpg %DATA_PATH%/image/pen2.jpg
adb push image/pen2.png %DATA_PATH%/image/pen2.png
adb push image/metall16bit.bmp %DATA_PATH%/image/metall16bit.bmp
adb push shader/vs.txt %DATA_PATH%/shader/vs.txt
adb push shader/ps.txt %DATA_PATH%/shader/ps.txt
adb push shader/vs-mesh.txt %DATA_PATH%/shader/vs-mesh.txt
adb push shader/ps-mesh.txt %DATA_PATH%/shader/ps-mesh.txt
adb push meshes/bat.hqmesh %DATA_PATH%/meshes/bat.hqmesh
adb push meshes/bat.hqanimation %DATA_PATH%/meshes/bat.hqanimation
adb push meshes/Bat_Albedo.jpg  %DATA_PATH%/meshes/Bat_Albedo.jpg
adb push meshes/tiny.hqmesh %DATA_PATH%/meshes/tiny.hqmesh
adb push meshes/tiny.hqanimation %DATA_PATH%/meshes/tiny.hqanimation
adb push meshes/Tiny_skin.bmp  %DATA_PATH%/meshes/Tiny_skin.bmp
adb push script/effects.script %DATA_PATH%/script/effects.script
adb push script/resourcesGLES.script %DATA_PATH%/script/resourcesGLES.script
adb push script/resourcesCommon.script %DATA_PATH%/script/resourcesCommon.script

popd
