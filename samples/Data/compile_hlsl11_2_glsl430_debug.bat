@echo off

set PREDEFINED=/DUSE_GL_WAY
set TOOLS_FOLDER=%~dp0..\..\utilities\HQShaderCompiler\Netbeans\HQEngineShaderCompiler
 
set cmd=%TOOLS_FOLDER%\fxc /nologo %1 /Tvs_5_0 /Zpc /Od /EVS %PREDEFINED% /Fo%~dp1%~n1.hlslv.bytecode
echo.
echo %cmd%
%cmd% || set error=1
set cmd=call %~dp0compile_hlsl11bytecode_2_glsl430.bat %~dp1%~n1.hlslv.bytecode %~dp1%~n1.glslv
echo.
echo %cmd%
%cmd% || set error=1

set cmd=%TOOLS_FOLDER%\fxc /nologo %1 /Tps_5_0 /Zpc /Od /EPS %PREDEFINED% /Fo%~dp1%~n1.hlslf.bytecode
echo.
echo %cmd%
%cmd% || set error=1
set cmd=call %~dp0compile_hlsl11bytecode_2_glsl430.bat %~dp1%~n1.hlslf.bytecode %~dp1%~n1.glslf
echo.
echo %cmd%
%cmd% || set error=1
exit /b