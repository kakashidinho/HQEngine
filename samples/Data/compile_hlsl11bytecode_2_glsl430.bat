@echo off

set TOOLS_FOLDER=%~dp0..\..\utilities\HQShaderCompiler\Netbeans\HQEngineShaderCompiler
 
set cmd=%TOOLS_FOLDER%\HQEXT_hlslasm2glsl -version 430 -o %2 %1
echo.
echo %cmd%
%cmd% || set error=1