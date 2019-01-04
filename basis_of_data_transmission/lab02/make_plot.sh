#!/usr/bin/gnuplot
reset

set terminal png
set grid
set style data lines

do for [i=1:7] {
    set output 'plot'.i.'_lin.png'
    unset logscale x
    plot 'plot'.i.'_ftw.csv'

    set output 'plot'.i.'_log.png'
    set logscale x
    plot 'plot'.i.'_ftw.csv'

    set output 'plot'.i.'.png'
    unset logscale x
    plot 'plot'.i.'.csv'
}
