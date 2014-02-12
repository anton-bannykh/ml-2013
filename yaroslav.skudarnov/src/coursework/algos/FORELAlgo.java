package coursework.algos;

import coursework.points.Cluster;
import coursework.points.Partition;
import coursework.points.Point;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by yaroslav on 31.01.14.
 */
public class FORELAlgo implements Algo {
    List<ArrayList<Double>> distances;
    final Random rnd = new Random();
    int bucketsQuantity;

    @Override
    public Partition clusterize(Cluster c) {
        Cluster set = new Cluster(c);

        double maxDistance = init(set);

        double radius = findRadius(maxDistance);

        return findPartition(set, radius);
    }

    private Partition findPartition(Cluster set, double radius) {
        Partition partition = new Partition();

        while (!set.isEmpty()) {
            Point[] points = set.toArray(new Point[set.size()]);
            Point centroid = points[rnd.nextInt(points.length)];
            Cluster oldSphere = new Cluster();

            Cluster newSphere = findCluster(points, radius, centroid);

            while (!newSphere.equals(oldSphere)) {
                oldSphere = newSphere;
                centroid = oldSphere.centroid();
                newSphere = findCluster(points, radius, centroid);
            }

            partition.put(centroid, newSphere);

            set.removeAll(newSphere);
        }

        return partition;
    }

    private Cluster findCluster(Point[] points, double radius, Point centroid) {
        Cluster c = new Cluster();

        for (Point p : points) {
            if (p.distance(centroid) < radius) {
                c.add(p);
            }
        }

        return c;
    }

    private double findRadius(double maxDistance) {
        int distancesBuckets[] = new int[bucketsQuantity];

        for (ArrayList<Double> l : distances) {
            for (Double d : l) {
                ++distancesBuckets[((int) (d * (bucketsQuantity - 1) / maxDistance))];
            }
        }

        int r = (int) (bucketsQuantity * 0.2);

        for (int i = r + 1; i < (int) (bucketsQuantity * 0.6); ++i) { // One more heuristic here
            if (distancesBuckets[i] < distancesBuckets[r]) {
                r = i;
            }
        }

        return r * maxDistance / bucketsQuantity;
    }

    private double init(Cluster set) {
        bucketsQuantity = set.size(); //just some empiric assumption
        distances = new ArrayList<ArrayList<Double>>();

        double maxDistance = Double.MIN_VALUE;

        for (Point p : set) {
            ArrayList<Double> tmp = new ArrayList<Double>();

            for (Point p1 : set) {
                tmp.add(p.distance(p1));

                if (p.distance(p1) > maxDistance) {
                    maxDistance = p.distance(p1);
                }
            }

            distances.add(tmp);
        }

        return maxDistance;
    }
}