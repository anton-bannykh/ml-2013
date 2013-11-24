#!/bin/sh

mkdir -p out/cancer/perceptron
javac src/cancer/*.java src/cancer/perceptron/*.java
cp src/cancer/*.class out/cancer/
cp src/cancer/perceptron/*.class out/cancer/perceptron
cd out
java cancer/perceptron/Perceptron
