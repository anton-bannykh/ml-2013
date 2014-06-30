package coursework.points;

import java.util.Arrays;

/**
 * Created by yaroslav on 31.01.14.
 */
public class Point {
    public final int DIMENSION;
    public double[] coords;

    Point(int _DIMENSTION) {
        DIMENSION = _DIMENSTION;
        coords = new double[DIMENSION];
    }

    public Point(double[] _coords) {
        DIMENSION = _coords.length;
        coords = Arrays.copyOf(_coords, DIMENSION);
    }

    public double distance(Point p) {
        double result = 0;

        if (p.DIMENSION != DIMENSION) {
            return Double.NaN;
        }

        for (int i = 0; i < DIMENSION; ++i) {
            result += (p.coords[i] - coords[i]) * (p.coords[i] - coords[i]);
        }

        return Math.sqrt(result);
    }

    @Override
    public int hashCode() {
        int result = DIMENSION;
        result = 31 * result + Arrays.hashCode(coords);
        return result;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Point)) {
            return false;
        }

        if (o == null) {
            return false;
        }

        Point tmp = (Point) o;

        return Arrays.equals(coords, tmp.coords) && DIMENSION == tmp.DIMENSION;
    }

    @Override
    public String toString() {
        return Arrays.toString(coords);
    }
}