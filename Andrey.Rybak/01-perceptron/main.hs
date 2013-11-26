import System.IO
import Data.List.Split
import qualified Data.List as L
import qualified Data.Vector as V

type FPType = Double

diagnosisFromChar :: Char -> Int
diagnosisFromChar x = case x of
  'M' -> 1
  'B' -> -1
  c   -> error $ "diagnosisFromChar : " ++ [c]

data DataSetItem = DataSetItem { diagnosis :: Int, attr :: V.Vector FPType}

proportion :: Double
proportion = 0.1
divide :: [a] -> ([a], [a])
divide as = splitAt (round(proportion * (fromIntegral n))) as where
  n = L.length as
                     
fromVector :: V.Vector String -> DataSetItem
fromVector v = DataSetItem ((diagnosisFromChar . L.head . ((flip (V.!)) 1)) v)
               (((V.map read). V.tail . V.tail) v)

map4 :: (a, a, a, a) -> (a -> b) -> (b, b, b, b)
map4 (x, y, z, t) f = (f x, f y, f z, f t)

fromStats :: (Int, Int, Int, Int) -> Int -> (FPType, FPType, FPType)
fromStats (fp, tp, fn, _) n = (p, r, e) where
  p = one * (fromIntegral tp) / (fromIntegral (tp + fp))
  r = (fromIntegral tp) / (fromIntegral (tp + fn))
  e = (fromIntegral (fp + fn)) / (fromIntegral n)
  one :: FPType
  one = 1.0

test :: [DataSetItem] -> V.Vector FPType -> (Int, Int, Int, Int)
test ts w = testAUX 0 0 0 0 ts where
  testAUX fp tp fn tn [] = (fp, tp, fn, tn)
  testAUX fp tp fn tn (x : xs) = if (diagnosis x) == 1
                                 then if ans ==  1 then testAUX fp (tp + 1) fn tn xs
                                                   else testAUX fp tp (fn + 1) tn xs
                                 else if ans == -1 then testAUX fp tp fn (tn + 1) xs
                                                   else testAUX (fp + 1) tp fn tn xs where
                                                        ans = classify w (attr x)

trainIterations :: Int
trainIterations = 1000
train :: [DataSetItem] -> V.Vector FPType
train ds = trainW w0 where
  dimensions = (V.length . attr) (L.head ds)
  w0 = V.replicate dimensions (0.0 :: FPType)
  trainW w = (last . (take trainIterations) . (L.iterate nextW)) w
  nextW w = nextWaux w ds
  nextWaux w [] = w
  nextWaux w (x : xs) = nextWaux (if (classify w (attr x)) /= (diagnosis x) then add w (diagnosis x) (attr x) else w) xs where
    add a y b = V.zipWith (+) a (V.map (* (fromIntegral y)) b)

classify :: V.Vector FPType -> V.Vector FPType -> Int
classify a b = if (V.foldr1 (+) (V.zipWith (*) a b)) >= 0 then 1 else -1

output :: String -> IO ()
output content = do 
                 putStrLn $ "Dataset size = " ++ show n
                 putStrLn $ "precision " ++ show p
                 putStrLn $ "recall " ++ show r
                 putStrLn $ "error " ++ show e where
  n :: Int
  n = L.length array
  ls :: [String]
  ls = lines content
  array :: [V.Vector String]
  array = map ((V.fromListN 32) . splitOn ",") ls
  dataset :: [DataSetItem]
  dataset = map fromVector array
  (testSet, trainSet) = divide dataset
  stats = test testSet (train trainSet)
  (p, r, e) = fromStats stats n

main :: IO ()
main = do
    file <- readFile "../data/wdbc.data"
    output file
