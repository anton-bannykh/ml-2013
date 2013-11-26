module WDBC.Writer (outputStats) where

import WDBC.Core
import qualified System.IO as IO (putStrLn)

fromStats :: (Int, Int, Int, Int) -> Int -> (FPType, FPType, FPType)
fromStats (fp, tp, fn, _) n = (p, r, e) where
  p = one * (fromIntegral tp) / (fromIntegral (tp + fp))
  r = (fromIntegral tp) / (fromIntegral (tp + fn))
  e = (fromIntegral (fp + fn)) / (fromIntegral n)
  one :: FPType
  one = 1.0

outputStats :: Stats -> Int -> IO ()
outputStats stats n = do
  IO.putStrLn $ "Dataset size = " ++ show n
  IO.putStrLn $ "precision " ++ show p
  IO.putStrLn $ "recall " ++ show r
  IO.putStrLn $ "error " ++ show e
    where
    (p, r, e) = fromStats stats n
 
