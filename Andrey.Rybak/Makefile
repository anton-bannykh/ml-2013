HC=ghc -O9 -i$(OBJ)
HCB=$(HC) -outputdir $(BIN)
HCO=$(HC) -c -outputdir $(OBJ)

BIN=bin
OBJ=obj
OBJW=$(OBJ)/WDBC
DATA=data

RM=rm -f 2> /dev/null
MD=mkdir -p

MDO=$(MD) $(OBJ)

main : main.hs $(DATA)/wdbc.data $(OBJ)/Perceptron.o $(OBJ)/WDBC.o
	$(MD) $(BIN)
	$(HCB) main.hs -o $(BIN)/main

$(OBJ)/Perceptron.o : Perceptron.hs $(OBJ)/WDBC/Core.o
	$(MDO)
	$(HCO) Perceptron.hs

clean :
	$(RM) -r $(BIN)
	$(RM) -r $(OBJ)
	$(RM) *.o
	$(RM) *.hi

$(OBJ)/WDBC.o : WDBC.hs $(OBJW)/Core.o $(OBJW)/Reader.o $(OBJW)/Writer.o
	$(MDO)/WDBC
	$(MD) $(OBJ)/WDBC
	$(HCO) WDBC.hs

$(OBJ)/WDBC/Core.o : WDBC/Core.hs
	$(MDO)/WDBC
	$(HCO) WDBC/Core.hs

$(OBJ)/WDBC/Reader.o : WDBC/Reader.hs
	$(MDO)/WDBC
	$(HCO) WDBC/Reader.hs

$(OBJ)/WDBC/Writer.o : WDBC/Writer.hs
	$(MDO)/WDBC
	$(HCO) WDBC/Writer.hs
