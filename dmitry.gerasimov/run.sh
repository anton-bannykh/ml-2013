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
    options=("lab-perceptron" "lab-svm" "lab-svm-smo" "lab-logistic" "lab-neural" "quit")
    select opt in "${options[@]}"
    do
        choice=$opt
	break
    done
fi


case $choice in
    "lab-perceptron")
        echo "Running $choice"
        PYTHONPATH=$WD:$PYTHONPATH python3 $choice/main.py
        ;;
    "lab-svm")
        echo "Running $choice"
        PYTHONPATH=$WD:$PYTHONPATH python3 $choice/main.py
        ;;
    "lab-svm-smo")
        echo "Running $choice"
        PYTHONPATH=$WD:$PYTHONPATH python3 $choice/main.py
        ;;
    "lab-logistic")
        echo "Running $choice"
        PYTHONPATH=$WD:$PYTHONPATH python3 $choice/main.py
        ;;
    "lab-neural")
        echo "Running $choice"
        PYTHONPATH=$WD:$PYTHONPATH python3 $choice/main.py
	;;
    "Quit")
        ;;
    *)
        echo "Invalid option"
        ;;
esac
