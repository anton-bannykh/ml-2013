package coursework.testing;

import coursework.algos.FORELAlgo;
import coursework.algos.KMeansAlgo;
import coursework.algos.KMeansPlusPlusAlgo;
import coursework.points.Cluster;
import coursework.points.Partition;
import coursework.points.Point;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Locale;
import java.util.Scanner;

/**
 * Created by yaroslav on 03.02.14.
 */
public class TestAlgos {
    /*private static final int POINTS = 6;
    private static final int DIMS = 2;*/
    private static final int POINTS = 40;
    private static final int DIMS = 4;

    private static Cluster readData(String filename) throws FileNotFoundException {
        Scanner sc = new Scanner(new File(filename));
        sc.useLocale(Locale.US);
        Cluster c = new Cluster();

        for (int i = 0; i < POINTS; ++i) {
            double[] coords = new double[DIMS];

//            System.out.println(sc.next());

            for (int j = 0; j < DIMS; ++j) {
                coords[j] = sc.nextDouble();
            }

            c.add(new Point(coords));
        }

        return c;
    }

    private static void writeResult(Partition p, String filename) throws FileNotFoundException {
        PrintWriter pw = new PrintWriter(new File(filename));

        pw.print(p);

        int i = 0;

        for (Cluster c : p.values()) {
            if (c.size() <= 1) {
                ++i;
            }
        }

        pw.println(p.quality() * p.size() / (p.size() - i)); //penalty for small clusters

        pw.close();
    }

    private static void testKMeans(int n, String filename) throws FileNotFoundException {
        Partition p = new KMeansAlgo(n).clusterize(readData(filename));

        writeResult(p, filename + "-" + n + "-Means");
    }

    private static void testKMeansPlusPlus(int n, String filename) throws FileNotFoundException {
        Partition p = new KMeansPlusPlusAlgo(n).clusterize(readData(filename));

        writeResult(p, filename + "-" + n + "-Means++");
    }

    private static void testFOREL(String filename) throws FileNotFoundException {
        Partition p = new FORELAlgo().clusterize(readData(filename));

        writeResult(p, filename + "-" + "FOREL");
    }

    public static void main(String args[]) throws FileNotFoundException {
        final String directoryName = "data\\";
        File dir = new File(directoryName);

        if (!dir.exists()) {
            dir.mkdir();
        }

        final String filename = directoryName + "4 dims, 4 gaussians, nextDouble(3), nextDouble(3), 6";

        testKMeans(4, filename);
        testKMeansPlusPlus(4, filename);
        testFOREL(filename);
    }
}