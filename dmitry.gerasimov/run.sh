#!/bin/bash

if [ ! -d "common" ]
then
    echo "[ERROR] haven't found 'common' directory"
    echo "[ERROR] please run the script from the 'dmitry.gerasimov' directory"
    exit 1
fi

WD="`pwd`/common"

choice=""
if [[ "$#" -ne 0 ]]
then
    choice=$1
else
    echo "Please enter your choice (Ctrl-C to abort):"
    options=("lab-perceptron" "lab-svm" "Quit")
    select opt in "${options[@]}"
    do
        choice=$opt
    done
fi


case $choice in
    "lab-perceptron")
        echo "Running $choice"
        PYTHONPATH=$WD:$PYTHONPATH python3 $choice/main.py
        ;;
    "lab-svn")
        echo "Running $choice"
        PYTHONPATH=$WD:$PYTHONPATH python3 $choice/main.py
        ;;
    "Quit")
        ;;
    *)
        echo "Invalid option"
        ;;
esac
