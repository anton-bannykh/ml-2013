import WDBC
import Perceptron

main :: IO ()
main = do
  putStrLn "Perceptron :"
  file <- readFile "data/wdbc.data"
  printTest file perceptron
