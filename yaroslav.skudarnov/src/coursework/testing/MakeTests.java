package coursework.testing;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Random;

/**
 * Created by yaroslav on 03.02.14.
 */
public class MakeTests {
    final static Random rnd = new Random();

    public static void generate4Gaussians() throws FileNotFoundException {
        final String directoryName = "data\\";
        File dir = new File(directoryName);

        if (!dir.exists()) {
            dir.mkdir();
        }

        PrintWriter pw = new PrintWriter(new File(directoryName + "4 dims, 4 gaussians, nextDouble(3), nextDouble(3), 6"));

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 10; ++j) {
                for (int k = 0; k < 4; ++k) {
                    if (i != k) {
                        pw.print(rnd.nextGaussian() * (3 * rnd.nextDouble()) + " ");
                    } else {
                        pw.print(rnd.nextGaussian() * (3 * rnd.nextDouble()) + 6 + " ");
                    }
                }

                pw.println();
            }
        }

        pw.close();
    }

    public static void main(String args[]) throws FileNotFoundException {
        generate4Gaussians();
    }
}