#!/usr/bin/gnuplot

set xlabel "Rabbits"
set ylabel "Wolves"
set grid ytics lt 0 lw 1 lc rgb "#bbbbbb"
set grid xtics lt 0 lw 1 lc rgb "#bbbbbb"
set autoscale
set terminal postscript portrait enhanced mono dashed lw 1 'Helvetica' 14
set style line 1 lt 1 lw 3 pt 3 linecolor rgb "red"
set output 'out.eps'
plot 'data.txt' using 1:2 w points title "steps"

