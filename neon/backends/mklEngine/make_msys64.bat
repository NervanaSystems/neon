set WIN_BUILD=1
set MKLROOT=mklml_win_2018.0.1.20171227
wget https://github.com/01org/mkl-dnn/releases/download/v0.12/%MKLROOT%.zip
pacman -S --needed unzip
unzip %MKLROOT%.zip
make
copy "%MKLROOT%\lib\mklml.dll"
ren *.so *.dll
rmdir /q /s %MKLROOT%
rm %MKLROOT%.zip
