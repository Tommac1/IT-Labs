#!/usr/bin/gnuplot
reset

set terminal png
set grid
set style data lines

set output 'plot0.png'
plot 'plot0.csv'

set output 'plot1.png'
set logscale x
plot 'plot1.csv'

set output 'plot2.png'
unset logscale x
plot 'plot2.csv'

set output 'plot3.png'
plot 'plot3.csv'

set output 'plot4.png'
set yrange [-1:2]
plot 'plot4.csv'
unset yrange

set output 'plot5.png'
plot 'plot5.csv'

set output 'plot6.png'
set logscale x
plot 'plot6.csv'

set output 'plot7.png'
unset logscale x
plot 'plot7.csv'

set output 'plot8.png'
plot 'plot8.csv'

set output 'plot9.png'
set yrange [-1:2]
plot 'plot9.csv'
unset yrange
