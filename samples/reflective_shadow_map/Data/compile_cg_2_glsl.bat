@echo off
 
set cmd=%~dp0HQEXT_cg2glsl -profile glslv -entry VS -o %2 -version 120 %1
echo.
echo %cmd%
%cmd% || set error=1

set cmd=%~dp0HQEXT_cg2glsl -profile glslf -entry PS -o %3 -version 120 %1
echo %cmd%
%cmd% || set error=1
exit /b