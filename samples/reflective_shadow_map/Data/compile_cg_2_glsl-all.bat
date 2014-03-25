@echo off

set TOOLS_FOLDER=%~dp0..\..\..\tools\HQShaderCompiler\Netbeans\HQEngineShaderCompiler
 
pushd %~dp0 
call :Compile depth-pass.cg
call :Compile noise_decoding.cg
call :Compile lowres-pass.cg
call :Compile final-gathering.cg

popd

endlocal
exit /b

:Compile
set cmd=%TOOLS_FOLDER%\HQEXT_cg2glsl -profile glslv -entry VS -o %~n1-compiled-cg.glslv -version 120 %1
echo.
echo %cmd%
%cmd% || set error=1

set cmd=%TOOLS_FOLDER%\HQEXT_cg2glsl -profile glslf -entry PS -o %~n1-compiled-cg.glslf -version 120 %1
echo %cmd%
%cmd% || set error=1
exit /b