@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=python -msphinx
)
set SOURCEDIR=.
set BUILDDIR=../../SilQ-documentation
set SPHINXPROJ=SilQ

if "%1" == "" goto help
af
%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The Sphinx module was not found. Make sure you have Sphinx installed,
	echo.then set the SPHINXBUILD environment variable to point to the full
	echo.path of the 'sphinx-build' executable. Alternatively you may add the
	echo.Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)


if "%1" == "gh-pages" (
    echo.Updating gh-pages
    set currentdir=%cd%
    echo current dir is %currentdir%
    cd %SOURCEDIR%/..
    git checkout gh-pages
    copy /Y %BUILDDIR/html ./
    rem git add -A
    cd %currentdir%
    goto end
)

rem sphinx-apidoc -o _modules ../silq

rem %SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

rem goto end

rem :help
rem %SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
popd
