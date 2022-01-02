#include <cmath>
#include <sciplot/sciplot.hpp>
using namespace sciplot;

int main() {
      std::vector<double> x = {0, 1, 2, 3, 4, 5, 6, 7};    
      std::vector<double> y = {0, 1, 2, 3, 4, 5, 6, 7};

  //  double x = 5.0;
  //  double y = 5.0;
  
    // Create a Plot object
    Plot plot;

    // Set the legend
    plot.legend().hide();

    // Set the x and y labels
    plot.xlabel("x");
    plot.ylabel("y");

    // Add values to plot
    plot.drawBoxes(x, y)
        .fillSolid()
        .fillColor("green")
        .fillIntensity(0.5);

    // Adjust the relative width of the boxes
    plot.boxWidthRelative(0.75);

    // Show the plot in a pop-up window
    plot.show();
    
    return 0;
}
