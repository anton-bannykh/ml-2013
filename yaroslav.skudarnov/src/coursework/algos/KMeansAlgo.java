package coursework.algos;

import coursework.points.Cluster;
import coursework.points.Partition;
import coursework.points.Point;

import java.util.Random;
import java.util.Set;

/**
 * Created by yaroslav on 31.01.14.
 */
public class KMeansAlgo implements Algo {
    final int CLUSTERS;

    Partition oldPartition, newPartition;
    Point[] points;
    Random rnd = new Random();

    public KMeansAlgo(int _CLUSTERS) {
        CLUSTERS = _CLUSTERS;
    }

    protected Point getNewRandomCentroid() {
        Point p;

        do {
            p = points[rnd.nextInt(points.length)];
        } while (oldPartition.containsKey(p));

        return p;
    }

    @Override
    public Partition clusterize(Cluster set) {
        initPartitions(set);

        initCentroids();

        doAlgo();

        return oldPartition;
    }

    protected void doAlgo() {
        newPartition = new Partition(oldPartition);

        reCalcClusters();

        do {
            oldPartition = new Partition(newPartition);

            newPartition = new Partition();

            reCalcCentroids();

            reCalcClusters();
        } while (!oldPartition.equals(newPartition));
    }

    protected void initPartitions(Cluster set) {
        oldPartition = new Partition();

        if (CLUSTERS > set.size()) {
            throw new RuntimeException("There are more clusters than points!");
        }

        points = set.toArray(new Point[set.size()]);
    }

    protected void initCentroids() {
        for (int i = 0; i < CLUSTERS; ++i) {
            Point p = getNewRandomCentroid();

            oldPartition.put(p, new Cluster());
        }
    }

    protected void reCalcClusters() {
        Set<Point> centroids = newPartition.keySet();

        Point nearest = null;

        for (Point p : points) {
            double distance = Double.MAX_VALUE;

            for (Point centroid : centroids) {
                double tmp;

                if ((tmp = p.distance(centroid)) < distance) {
                    distance = tmp;
                    nearest = centroid;
                }
            }

            newPartition.get(nearest).add(p);
        }
    }

    protected Set<Point> reCalcCentroids() {
        Set<Point> centroids = oldPartition.keySet();

        for (Point centroid : centroids) {
            int dim = centroid.DIMENSION;

            double coords[] = new double[dim];

            if (!oldPartition.get(centroid).isEmpty()) {
                newPartition.put(oldPartition.get(centroid).centroid(), new Cluster());
            } else {
                Point p;

                do {
                    p = points[rnd.nextInt(points.length)];
                } while (oldPartition.containsKey(p));

                newPartition.put(p, new Cluster());
            }
        }

        return centroids;
    }
}