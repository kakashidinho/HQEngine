@echo off

set TOOLS_FOLDER=%~dp0..\..\..\utilities\HQShaderCompiler\Netbeans\HQEngineShaderCompiler
 
set cmd=%TOOLS_FOLDER%\cgc -profile vs_4_0 -entry VS -DHQEXT_CG -o %2  %1
echo.
echo %cmd%
%cmd% || set error=1

set cmd=%TOOLS_FOLDER%\cgc -profile ps_4_0 -entry PS -DHQEXT_CG -o %3  %1
echo %cmd%
%cmd% || set error=1
exit /b