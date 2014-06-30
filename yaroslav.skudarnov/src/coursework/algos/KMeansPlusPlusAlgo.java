package coursework.algos;

import coursework.points.Cluster;
import coursework.points.Point;

/**
 * Created by yaroslav on 06.02.14.
 */
public class KMeansPlusPlusAlgo extends KMeansAlgo {
    public KMeansPlusPlusAlgo(int _CLUSTERS) {
        super(_CLUSTERS);
    }

    private double minDistanceToCentroid(Point p) {
        double distance = Double.MAX_VALUE;

        for (Point centroid : oldPartition.keySet()) {
            double tmp;

            if ((tmp = p.distance(centroid)) < distance) {
                distance = tmp;
            }
        }

        return distance;
    }

    @Override
    protected void initCentroids() {
        oldPartition.put(getNewRandomCentroid(), new Cluster());

        while (oldPartition.size() < CLUSTERS) {
            double sumOfDistSqs = 0;

            for (Point p : points) {
                double distance = minDistanceToCentroid(p);

                sumOfDistSqs += distance * distance;
            }

            Point potentialCentroid = null;

            do {
                double sumOfDistSqsToNewCentroid = rnd.nextDouble() * sumOfDistSqs;

                double tmp1 = 0;

                for (Point p : points) {
                    double distance = minDistanceToCentroid(p);

                    tmp1 += distance * distance;

                    if (tmp1 > sumOfDistSqsToNewCentroid) {
                        potentialCentroid = p;
                        break;
                    }
                }
            } while (oldPartition.containsKey(potentialCentroid));

            oldPartition.put(potentialCentroid, new Cluster());
        }
    }
}