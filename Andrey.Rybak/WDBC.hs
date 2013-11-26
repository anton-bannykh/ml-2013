module WDBC (printTest) where

import qualified Data.List as L

import WDBC.Core
import WDBC.Reader
import WDBC.Writer

printTest :: String -> ([DataSetItem] -> Stats) -> IO ()
printTest file method = outputStats stats n where
  n = L.length dataset
  dataset = readDataSet file
  stats = method dataset
