package cancer;

import java.util.ArrayList;

/**
 * User: Skudarnov Yaroslav
 * Date: 03.11.13
 * Time: 23:41
 */
class Samples {
    static class TestSample {
        private TestSample() {
            result = new ArrayList<Integer>();
            features = new ArrayList<ArrayList<Double>>();
        }

        public static final TestSample testSample = new TestSample();

        ArrayList<Integer> result;
        ArrayList<ArrayList<Double>> features;
    }

    static class TrainingSample {
        private TrainingSample() {
            result = new ArrayList<Integer>();
            features = new ArrayList<ArrayList<Double>>();
        }

        public static final TrainingSample trainingSample = new TrainingSample();

        ArrayList<Integer> result;
        ArrayList<ArrayList<Double>> features;
    }

    TestSample testSample;
    TrainingSample trainingSample;

    Samples() {
        testSample = TestSample.testSample;
        trainingSample = TrainingSample.trainingSample;
    }
}