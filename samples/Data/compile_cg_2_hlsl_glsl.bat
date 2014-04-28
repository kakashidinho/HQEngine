@echo off

set DEST=%~dp1

call :Compile %1

endlocal
exit /b

:Compile
call %~dp0compile_cg_2_hlsl.bat %1 %DEST%%~n1-compiled-cg.hlslv %DEST%%~n1-compiled-cg.hlslf
call %~dp0compile_cg_2_glsl.bat %1 %DEST%%~n1-compiled-cg.glslv %DEST%%~n1-compiled-cg.glslf
exit /b
