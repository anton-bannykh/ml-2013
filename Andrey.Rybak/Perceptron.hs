module Perceptron (perceptron) where

import qualified Data.List as L
import qualified Data.Vector as V

import WDBC.Core

proportion :: Double
proportion = 0.1
divide :: [a] -> ([a], [a])
divide as = splitAt (round(proportion * (fromIntegral n))) as where
  n = L.length as
                     
trainIterations :: Int
trainIterations = 1000
train :: [DataSetItem] -> V.Vector FPType
train dataset = trainW w0 where
  dimensions = (V.length . attr) (L.head dataset)
  w0 = V.replicate dimensions (0.0 :: FPType)
  trainW w = (L.last . (L.take trainIterations) . (L.iterate nextW)) w
  nextW w = nextWaux w dataset
  nextWaux w [] = w
  nextWaux w (x : xs) = nextWaux
    (if (classify w (attr x)) /= (diagnosis x)
     then add w (diagnosis x) (attr x) else w) xs
    where
    add a y b = V.zipWith (+) a (V.map (* (fromIntegral y)) b)

classify :: V.Vector FPType -> V.Vector FPType -> Int
classify a b = if (V.foldr1 (+) (V.zipWith (*) a b)) >= 0 then 1 else -1

perceptron :: [DataSetItem] -> Stats
perceptron dataset = stats where
  (testSet, trainSet) = divide dataset
  w = (train trainSet)
  stats = runTest testSet (classify w)
