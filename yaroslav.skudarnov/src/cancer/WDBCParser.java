package cancer;

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
class WDBCParser {
    private final String WDBCURL;
    private ArrayList<String> data;
    private Samples samples;

    WDBCParser(String URL) {
        WDBCURL = URL;
    }

    private void getData() {
        try {
            URL url = new URL(WDBCURL);
            InputStream is = url.openStream();
            Scanner sc = new Scanner(is);

            data = new ArrayList<String>();

            while (sc.hasNext()) {
                data.add(sc.nextLine());
            }
        } catch (MalformedURLException e) {
            System.err.println("Wrong URL!");
            e.printStackTrace();
            System.exit(-1);
        } catch (IOException e) {
            System.err.println("Some error occured during reading a file from " + WDBCURL + ".");
            e.printStackTrace();
            System.exit(-1);
        }

        System.out.println("*****DATA DOWNLOADED*****");
        System.out.println();
    }

    public Samples parseWDBC(int percent) {
        getData();

        samples = new Samples();

        System.out.println("*****DATA IS BEING PARSED*****");
        System.out.println();

        getTestSample(percent);
        getTrainingSample(percent);

        System.out.println("*****DATA HAVE BEEN PARSED*****");
        System.out.println();

        return samples;
    }

    private void getTrainingSample(int percent) {
        int trainingSampleSize = data.size() * percent / 100;

        for (int i = 0; i < trainingSampleSize; ++i) {
            StringTokenizer st = new StringTokenizer(data.get(i), ",");

            st.nextToken();

            String diagnosis = st.nextToken();

            if (diagnosis.equals("M")) {
                samples.trainingSample.result.add(1);
            } else if (diagnosis.equals("B")) {
                samples.trainingSample.result.add(-1);
            } else {
                System.err.println("Wrong file format; get " + diagnosis + " instead of \"M\" or \"B\".");
                System.exit(-1);
            }

            samples.trainingSample.features.add(new ArrayList<Double>());

            while (st.hasMoreTokens()) {
                samples.trainingSample.features.get(i).add(Double.parseDouble(st.nextToken()));
            }
        }
    }

    private void getTestSample(int percent) {
        int samplesSize = data.size();
        int trainingSampleSize = samplesSize * percent / 100;
        int testSampleSize = samplesSize - trainingSampleSize;

        for (int i = 0; i < testSampleSize; ++i) {
            StringTokenizer st = new StringTokenizer(data.get(i + trainingSampleSize), ",");

            st.nextToken();

            String diagnosis = st.nextToken();

            if (diagnosis.equals("M")) {
                samples.testSample.result.add(1);
            } else if (diagnosis.equals("B")) {
                samples.testSample.result.add(-1);
            } else {
                System.err.println("Wrong file format; get " + diagnosis + " instead of \"M\" or \"B\".");
                System.exit(-1);
            }

            samples.testSample.features.add(new ArrayList<Double>());

            while (st.hasMoreTokens()) {
                samples.testSample.features.get(i).add(Double.parseDouble(st.nextToken()));
            }
        }
    }
}