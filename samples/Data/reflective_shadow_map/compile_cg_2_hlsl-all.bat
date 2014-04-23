@echo off

pushd %~dp0 
call :Compile depth-pass.cg
call :Compile noise_decoding.cg
call :Compile lowres-pass.cg
call :Compile final-gathering.cg

popd

endlocal
exit /b

:Compile
call compile_cg_2_hlsl.bat %1 %~n1-compiled-cg.hlslv %~n1-compiled-cg.hlslf
exit /b
