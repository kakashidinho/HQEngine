@echo off
set PATH=%~dp0..\..\..\utilities\Third-Party\GnuWin32\bin;%PATH%
set cmd=bison.exe  %*
echo %cmd%
%cmd% || set error=1