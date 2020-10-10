#!/bin/sh

find . -name "*.aux" -print0 | xargs -0 rm
find . -maxdepth 1 -name "*.bbl" -print0 | xargs -0 rm 
find . -maxdepth 1 -name "*.blg" -print0 | xargs -0 rm 
find . -maxdepth 1 -name "*.lof" -print0 | xargs -0 rm 
find . -maxdepth 1 -name "*.log" -print0 | xargs -0 rm 
find . -maxdepth 1 -name "*.out" -print0 | xargs -0 rm 
find . -maxdepth 1 -name "*.synctex.gz" -print0 | xargs -0 rm 
find . -maxdepth 1 -name "*.toc" -print0 | xargs -0 rm 
find . -maxdepth 1 -name "*.pdf" -print0 | xargs -0 rm
find . -maxdepth 1 -name "*.lol" -print0 | xargs -0 rm