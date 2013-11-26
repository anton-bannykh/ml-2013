import WDBC
import Perceptron

main :: IO ()
main = do
  putStrLn "Percetron :"
  file <- readFile "data/wdbc.data"
  printTest file perceptron
