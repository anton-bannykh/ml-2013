package coursework.points;

import java.util.HashSet;

/**
 * Created by yaroslav on 31.01.14.
 */

public class Cluster extends HashSet<Point> {
    public Cluster() {
        super();
    }

    public Cluster(Cluster c) {
        super(c);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Cluster)) {
            return false;
        }

        if (o == null) {
            return false;
        }

        Cluster c = (Cluster) o;

        return (containsAll(c)) && (c.containsAll(this));
    }

    @Override
    public String toString() {
        String tmp = "";

        for (Point point : this) {
            tmp += point + "\n";
        }

        return tmp;
    }

    public Point centroid() {
        int dim = iterator().next().DIMENSION;

        double[] coords = new double[dim];

        for (Point p : this) {
            for (int i = 0; i < p.DIMENSION; ++i) {
                coords[i] += p.coords[i];
            }
        }

        for (int i = 0; i < dim; ++i) {
            coords[i] /= size();
        }

        return new Point(coords);
    }
}