# This builds the test.cpp program.  It is a testing
# ground to see what build flags I need.  It is easier
# to edit this file than experiment using cmake.

# g++ -std=c++17 test.cpp -o test
g++ -std=c++17 -I/usr/local/include -L/usr/local/lib -o test test.cpp

