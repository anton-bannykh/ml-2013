package cancer.perceptron;

import cancer.WDBCParser;
import cancer.Samples;

import java.util.ArrayList;

/**
 * User: Skudarnov Yaroslav
 * Date: 04.11.13
 * Time: 16:53
 */
public class Perceptron {
    public static final int FEATURES = 30;
    public static int TESTSAMPLESIZE, TRAININGSAMPLESIZE;
    private static int truePositive, falsePositive, trueNegative, falseNegative;
    private static final int percent = 15;

    static ArrayList<Double> weights;

    public static void main(String args[]) {
        WDBCParser parser = new WDBCParser("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data");

        parser.parseWDBC(percent);

        init();
        train();
        test();
    }

    private static void init() {
        TESTSAMPLESIZE = Samples.testSample.features.size();
        TRAININGSAMPLESIZE = Samples.trainingSample.features.size();

        weights = new ArrayList<Double>(FEATURES);

        for (int i = 0; i < FEATURES; ++i) {
            weights.add((double) 0);
        }
    }

    private static void train() {
        boolean changing;

        ArrayList<Double> bestWeights = (ArrayList<Double>) weights.clone();
        int bestClassified = 0;
        int steps = 0;

        do {
            int tmp = 0;

            changing = false;

            for (int i = 0; i < TRAININGSAMPLESIZE; ++i) {
                if (wrongClassified(Samples.trainingSample.result.get(i), weights, Samples.trainingSample.features.get(i))) {
                    changing |= true;
                    weights = add(weights, mul(Samples.trainingSample.result.get(i), Samples.trainingSample.features.get(i)));
                }
            }

            for (int i = 0; i < TRAININGSAMPLESIZE; ++i) {
                if (wrongClassified(Samples.trainingSample.result.get(i), weights, Samples.trainingSample.features.get(i))) {
                    ++tmp;
                }
            }

            if (TRAININGSAMPLESIZE - tmp > bestClassified) {
                bestClassified = TRAININGSAMPLESIZE - tmp;
                bestWeights = (ArrayList<Double>) weights.clone();
            }

            System.out.println(bestClassified);

            if (++steps > TRAININGSAMPLESIZE * FEATURES) {
            //if (++steps > 3) {
                break;
            }
        } while (changing);

        weights = (ArrayList<Double>) bestWeights.clone();
    }

    private static boolean wrongClassified(int result, ArrayList<Double> weights, ArrayList<Double> features) {
        return result != algo(weights, features);
    }

    private static ArrayList<Double> add(ArrayList<Double> weights, ArrayList<Double> features) {
        for (int i = 0; i < FEATURES; ++i) {
            weights.set(i, weights.get(i) + features.get(i));
        }

        return weights;
    }

    private static ArrayList<Double> mul(int sign, ArrayList<Double> features) {
        ArrayList<Double> tmp = new ArrayList<Double>(FEATURES);

        for (int i = 0; i < FEATURES; ++i) {
            tmp.add(features.get(i) * sign);
        }

        return tmp;
    }

    private static int algo(ArrayList<Double> weights, ArrayList<Double> features) {
        if (scalar(weights, features) >= 0) {
            return 1;
        } else {
            return -1;
        }
    }

    private static Double scalar(ArrayList<Double> weights, ArrayList<Double> features) {
        double result = 0;

        for (int i = 0; i < FEATURES; ++i) {
            result += weights.get(i) * features.get(i);
        }

        return result;
    }

    private static void test() {
        for (int i = 0; i < TESTSAMPLESIZE; ++i) {
            if (wrongClassified(Samples.testSample.result.get(i), weights, Samples.testSample.features.get(i))) {
                if (Samples.testSample.result.get(i) == 1) {
                    ++falseNegative;
                } else {
                    ++falsePositive;
                }
            } else {
                if (Samples.testSample.result.get(i) == 1) {
                    ++truePositive;
                } else {
                    ++trueNegative;
                }
            }
        }

        double precision = (double) truePositive / (truePositive + falsePositive);
        double recall = (double) truePositive / (truePositive + falseNegative);

        System.out.println("***PRECISION: " + precision);
        System.out.println("***RECALL: " + recall);
        System.out.println("***F1-metric: " + 2 * precision * recall / (precision + recall));
        System.out.println("***FALSE POSITIVE: " + falsePositive);
        System.out.println("***FALSE NEGATIVE: " + falseNegative);
        System.out.println("***TRUE POSITIVE: " + truePositive);
        System.out.println("***TRUE NEGATIVE: " + trueNegative);
        System.out.println("***PERCENT: " + percent);
    }
}