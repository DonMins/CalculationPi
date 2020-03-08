# About app

The method of calculating the number π by the Monte Carlo method is reduced to the simplest enumeration of points in the area.
The essence of the calculation is that we take a square with side a = 2R, enter a circle of radius R.
And we start at random to put dots inside the square. Geometrically, the probability P1 of the point falling into the circle,
equal to the ratio of the area of ​​the circle and square:
P1 = S circle / S square = π / 4

The probability of a point falling into a circle can also be calculated after a numerical experiment even more simply:
count the number of points in the circle and divide them by the total number of points set:
P2 = NCircle / Npoints
Consequently:
π / 4 = Nfallen into the circle / Npoints;
π = 4 N falling into the circle / N points;


This application allows you to calculate the π number using CPU and GPU.

## Using
1. Install CUDA on your computer (required to have a video card from Nvidia).
2. Create new CUDA project in Visual Studio.
3. Copy this code.
4. Run the application.


## System configuration

| Name  | Values  |
|-------|---------|
| CPU  | AMD Ryzen 5 2600 Six-Core Processor 3.4 GHz (Turbo Boost 3.9 GHz) |
| RAM  | 16 GB DDR4 |
| GPU  | GIGABYTE GeForce GTX 550 Ti [GV-N550D5-1GI]  |
| OS   | Windows 10 64-bit  |

## Results

The average time in milliseconds for 5 measurements is presented in the table. Matrix elements are of type float.

| Total points| time CPU |  time GPU  | Acceleration | result π CPU | result π GPU  | difference from exact |
|-------------|----------|------------|--------------|--------------|---------------|-----------------------|
|    65536    | 1 ms     |0.31 ms     |    3.22      |   3.14276    |      3.14276  |      -0.00117123      |
|    131072   | 1 ms     |0.31 ms     |    3.22      |   3.14862    |   3.14862     |       -0.00703061     |
|    262144   | 2 ms     | 0.32 ms    |    6.09      |    3.1412    |    3.1412     |    0.000385166        |
|    524288   | 3 ms     | 0.38       |    7.86      |    3.14315   |   3.14315     |      -0.00156033      |
|    1048576  | 6 ms     |   0.83     |    7.21      |    3.14042   |3.14042        |    0.00117481         |
|    2097152   | 12 ms   | 1.3 ms     |    9.23      | 3.14146      | 3.14146       |     0.000127674       |
|    4194304   |   23 ms | 2.63 ms    |    8.73      |   3.14062    |3.14062        |   0.000971676         |
|    8388608   | 46 ms   | 4.81 ms    |    9.54      |   3.14189    | 3.14189       |     -0.000297665      |
|    16777216   | 91 ms  |  8.96 ms   |    10.14     |    3.14126   | 3.14126       |     0.000326038       |
|    33554432   | 183 ms | 17.09 ms   |    10.70     |   3.14145    |   3.14145     |     0.000139833       |

The most accurate version was obtained at 2097152 points.
