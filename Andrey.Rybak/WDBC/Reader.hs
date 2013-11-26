module WDBC.Reader (readDataSet) where

import WDBC.Core
import qualified Data.Vector as V
import qualified Data.List as L
import qualified Data.List.Split as LS

diagnosisFromChar :: Char -> Int
diagnosisFromChar x = case x of
  'M' -> 1
  'B' -> -1
  c   -> error $ "diagnosisFromChar : " ++ [c]
 
fromVector :: V.Vector String -> DataSetItem
fromVector v = DataSetItem
  ((diagnosisFromChar . L.head . flip (V.!) 1) v)
  ((V.map read. V.tail . V.tail) v)
     
readDataSet :: String -> [DataSetItem]
readDataSet file = dataset where
  xs = lines file
  array = map (V.fromListN 32 . LS.splitOn ",") xs
  dataset = map fromVector array
