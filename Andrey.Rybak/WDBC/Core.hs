module WDBC.Core
       (FPType,
        DataSetItem(..),
        Stats,
        runTest
        ) where

import qualified Data.Vector as V

type FPType = Double

data DataSetItem = DataSetItem { diagnosis :: Int, attr :: V.Vector FPType}

type Stats = (Int, Int, Int, Int)

runTest :: [DataSetItem] -> (V.Vector FPType -> Int) -> Stats
runTest ts p = testAUX 0 0 0 0 ts where
  testAUX fp tp fn tn [] = (fp, tp, fn, tn)
  testAUX fp tp fn tn (x : xs) =
    if (diagnosis x) == 1
    then
      if ans == 1
      then testAUX fp (tp + 1) fn tn xs
      else testAUX fp tp (fn + 1) tn xs
    else
      if ans == -1
      then testAUX fp tp fn (tn + 1) xs
      else testAUX (fp + 1) tp fn tn xs where ans = p (attr x)

