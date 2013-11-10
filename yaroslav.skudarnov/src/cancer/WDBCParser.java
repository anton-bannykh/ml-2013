package cancer;

import cancer.perceptron.Perceptron;

import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.StringTokenizer;

/**
 * User: Skudarnov Yaroslav
 * Date: 03.11.13
 * Time: 23:33
 */
public class WDBCParser {
    private final String WDBCURL;
    private ArrayList<String> data;
    private Samples samples;

    public WDBCParser(String URL) {
        WDBCURL = URL;
    }

    private void getData() {
        System.out.println("*****DATA IS BEING DOWNLOADED*****");
        System.out.println();

        try {
            URL url = new URL(WDBCURL);
            InputStream is = url.openStream();
            Scanner sc = new Scanner(is);

            data = new ArrayList<String>();

            while (sc.hasNext()) {
                data.add(sc.nextLine());
            }

            /*for (int i = 0; i < data.size(); ++i) {
                int j = (int) (Math.random() * data.size());

                String tmp = data.get(i);
                data.set(i, data.get(j));
                data.set(j, tmp);
            }*/
        } catch (MalformedURLException e) {
            System.err.println("Wrong URL!");
            e.printStackTrace();
            System.exit(-1);
        } catch (IOException e) {
            System.err.println("Some error occured during reading a file from " + WDBCURL + ".");
            e.printStackTrace();
            System.exit(-1);
        }

        System.out.println("*****DATA HAVE BEEN DOWNLOADED*****");
        System.out.println();
    }

    public Samples parseWDBC(int percent) {
        getData();

        samples = new Samples();

        Perceptron.TRAININGSAMPLESIZE = data.size() * percent / 100;
        Perceptron.TESTSAMPLESIZE = data.size() - Perceptron.TRAININGSAMPLESIZE;

        System.out.println("*****DATA IS BEING PARSED*****");
        System.out.println();

        getTrainingSample(percent);
        getTestSample(percent);

        System.out.println("*****DATA HAVE BEEN PARSED*****");
        System.out.println();

        return samples;
    }

    private void getTrainingSample(int percent) {
        for (int i = 0; i < Perceptron.TRAININGSAMPLESIZE; ++i) {
            StringTokenizer st = new StringTokenizer(data.get(i), ",");

            st.nextToken();

            String diagnosis = st.nextToken();

            if (diagnosis.equals("M")) {
                Samples.trainingSample.result.add(1);
            } else if (diagnosis.equals("B")) {
                Samples.trainingSample.result.add(-1);
            } else {
                System.err.println("Wrong file format; get " + diagnosis + " instead of \"M\" or \"B\".");
                System.exit(-1);
            }

            Samples.trainingSample.features.add(new ArrayList<Double>(Perceptron.FEATURES));

            while (st.hasMoreTokens()) {
                Samples.trainingSample.features.get(i).add(Double.parseDouble(st.nextToken()));
            }
        }
    }

    private void getTestSample(int percent) {
        int samplesSize = data.size();

        for (int i = 0; i < Perceptron.TESTSAMPLESIZE; ++i) {
            StringTokenizer st = new StringTokenizer(data.get(i + Perceptron.TRAININGSAMPLESIZE), ",");

            st.nextToken();

            String diagnosis = st.nextToken();

            if (diagnosis.equals("M")) {
                Samples.testSample.result.add(1);
            } else if (diagnosis.equals("B")) {
                Samples.testSample.result.add(-1);
            } else {
                System.err.println("Wrong file format; get " + diagnosis + " instead of \"M\" or \"B\".");
                System.exit(-1);
            }

            Samples.testSample.features.add(new ArrayList<Double>(Perceptron.FEATURES));

            while (st.hasMoreTokens()) {
                Samples.testSample.features.get(i).add(Double.parseDouble(st.nextToken()));
            }
        }
    }
}