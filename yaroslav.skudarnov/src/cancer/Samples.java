package cancer;

import java.util.ArrayList;

/**
 * User: Skudarnov Yaroslav
 * Date: 03.11.13
 * Time: 23:41
 */
public class Samples {
    public static class TestSample {
        private TestSample() {
            result = new ArrayList<Integer>();
            features = new ArrayList<ArrayList<Double>>();
        }

        public static final TestSample testSample = new TestSample();

        public ArrayList<Integer> result;
        public ArrayList<ArrayList<Double>> features;
    }

    public static class TrainingSample {
        private TrainingSample() {
            result = new ArrayList<Integer>();
            features = new ArrayList<ArrayList<Double>>();
        }

        public static final TrainingSample trainingSample = new TrainingSample();

        public ArrayList<Integer> result;
        public ArrayList<ArrayList<Double>> features;
    }

    public static TestSample testSample;
    public static TrainingSample trainingSample;

    Samples() {
        testSample = TestSample.testSample;
        trainingSample = TrainingSample.trainingSample;
    }
}