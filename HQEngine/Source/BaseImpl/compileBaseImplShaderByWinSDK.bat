@echo off
rem THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
rem ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
rem THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
rem PARTICULAR PURPOSE.
rem
rem Copyright (c) Microsoft Corporation. All rights reserved.

setlocal

echo You must set the Directory containing the fxc of Windows SDK to Environment variables

set error=0

set debug_release=0

if "%1"=="debug" (
	echo compile Release version of shaders with debug symbols
	set debug_release=1
)

set COMPILED_DIR=%~dp0compiledBaseImplShader
if not exist %COMPILED_DIR% mkdir %COMPILED_DIR%

call :CompileShader HQClearViewportShaderCodeD3D1x vs VS
call :CompileShader HQClearViewportShaderCodeD3D1x ps PS

call :CompileShader HQFFEmuShaderD3D1x vs VS _light_spec_tex /DUSE_LIGHTING /DUSE_SPECULAR /DUSE_TEXTURE
call :CompileShader HQFFEmuShaderD3D1x vs VS _light_spec_notex /DUSE_LIGHTING /DUSE_SPECULAR
call :CompileShader HQFFEmuShaderD3D1x vs VS _light_nospec_tex /DUSE_LIGHTING /DUSE_TEXTURE
call :CompileShader HQFFEmuShaderD3D1x vs VS _light_nospec_notex /DUSE_LIGHTING
call :CompileShader HQFFEmuShaderD3D1x vs VS _nolight_nospec_tex /DUSE_TEXTURE
call :CompileShader HQFFEmuShaderD3D1x vs VS _nolight_nospec_notex
call :CompileShader HQFFEmuShaderD3D1x ps PS _tex /DUSE_TEXTURE
call :CompileShader HQFFEmuShaderD3D1x ps PS _notex 

echo.

if %error% == 0 (
    echo Shaders compiled ok
) else (
    echo There were shader compilation errors!
)

endlocal
exit /b

rem %1=file %2=shader type %3=entry point %4=prefix(can be empty) %5-7=macros definations

:CompileShader

rem compile both debug and release versions and generate debug symbols

set fxc_deb=fxc /nologo %~dp0%1.hlsl /T%2_4_0_level_9_3 /Zpc /Zi /Od /E%3 %5 %6 %7 /Fh%COMPILED_DIR%\%1_%3%4_debug.h /Fd%COMPILED_DIR%\%1_%3%4_debug.pdb /Vn%1_%3%4

if %debug_release% == 1 (
	set fxc_rel=fxc /nologo %~dp0%1.hlsl /T%2_4_0_level_9_3 /Zpc /Zi /E%3 %5 %6 %7 /Fh%COMPILED_DIR%\%1_%3%4.h /Fd%COMPILED_DIR%\%1_%3%4.pdb /Vn%1_%3%4
) else (
	set fxc_rel=fxc /nologo %~dp0%1.hlsl /T%2_4_0_level_9_3 /Zpc /E%3 %5 %6 %7 /Fh%COMPILED_DIR%\%1_%3%4.h /Vn%1_%3%4
)

echo.
echo %fxc_deb%
%fxc_deb% || set error=1

echo.
echo %fxc_rel%
%fxc_rel% || set error=1
exit /b