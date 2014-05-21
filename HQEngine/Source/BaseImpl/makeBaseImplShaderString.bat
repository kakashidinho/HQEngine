@echo off
rem THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF
rem ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO
rem THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
rem PARTICULAR PURPOSE.
rem
rem Copyright (c) Microsoft Corporation. All rights reserved.

setlocal
set error=0

pushd %~dp0

set OUT_DIR=BaseImplShaderString
if not exist %OUT_DIR% mkdir %OUT_DIR%

call :MakeShaderString HQFFEmuShaderD3D1x hlsl

call :MakeShaderString HQFFEmuShaderGL glsl -no-windows-newline

popd

endlocal
exit /b

:MakeShaderString

set cmd=HQTextFileToCHeader %1.%2 %1 %OUT_DIR%\%1.h %3
echo.
echo %cmd%
%cmd%
exit /b