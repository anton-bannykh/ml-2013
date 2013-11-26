module WDBC.Core
       (FPType,
        DataSetItem(..),
        Stats,
        runTest,
        divide2,
        divide3
        ) where

import qualified Data.Vector as V
import qualified Data.List as L

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

divide3 :: Double -> Double -> [a] -> ([a], [a], [a])
divide3 p1 p2 as = (xs, ys, zs) where
  (xs, rest) = divide2 p1 as
  (ys, zs) = divide2 (p2 / (1.0 - p1)) rest

divide2 :: Double -> [a] -> ([a], [a])
divide2 p as = splitAt (round(p * (fromIntegral n))) as where
  n = L.length as
 
