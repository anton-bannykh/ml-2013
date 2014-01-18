#!/bin/bash

mkdir -p bin
ghc -O9 --make -outputdir bin main.hs -o bin/main -rtsopts
