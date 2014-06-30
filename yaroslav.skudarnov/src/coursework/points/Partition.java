package coursework.points;

import java.util.HashMap;

/**
 * Created by yaroslav on 02.02.14.
 */

public class Partition extends HashMap<Point, Cluster> {
    public Partition() {
        super();
    }

    public Partition(Partition p) {
        super(p);
    }

    public boolean equals(Object o) {
        if (!(o instanceof Partition)) {
            return false;
        }

        if (o == null) {
            return false;
        }

        Partition p = (Partition) o;

        for (Cluster c : values()) {
            if (!p.containsValue(c)) {
                return false;
            }
        }

        for (Cluster c : p.values()) {
            if (!containsValue(c)) {
                return false;
            }
        }

        return true;
    }

    @Override
    public String toString() {
        String tmp = "";

        for (Point centroid : keySet()) {
            tmp += centroid + "\n\n" + get(centroid) + "\n\n\n";
        }

        return tmp;
    }

    public double quality() {
        return getAverageDistanceInClusters() / getAverageDistanceBetweenClusters();
    }

    private double getAverageDistanceInClusters() {
        double tmp = 0;

        int i = 0;

        for (Point p : keySet()) {
            for (Point p1 : get(p)) {
                if (!p.equals(p1)) {
                    tmp += p.distance(p1);
                    ++i;
                }
            }
        }

        return tmp / i;
    }

    private double getAverageDistanceBetweenClusters() {
        double tmp = 0;

        int i = 0;

        for (Point p : keySet()) {
            for (Cluster c : values()) {
                if (!get(p).equals(c)) {
                    for (Point p1 : c) {
                        tmp += p.distance(p1);
                        ++i;
                    }
                }
            }
        }

        return tmp / i;
    }
}