% file: PGL_4_2_sims_2.mp
% created by Orbiter metapost interface
% creation date: Thu Dec 17 16:59:38 MST 2020

input boxes

verbatimtex
\documentclass[10pt]{article}
\usepackage[noBBpl]{mathpazo}
\usepackage{amsmath}
\usepackage{amssymb}
\begin{document}
\scriptsize
etex;
u=0.100000mm;
beginfig(1);
path p[];
draw (500u ,950u )--(500u ,850u );
draw (500u ,850u )--(300u ,750u );
draw (500u ,850u )--(700u ,750u );
draw (500u ,850u )--(900u ,750u );
draw (300u ,750u )--(60u ,649.9u );
draw (300u ,750u )--(240u ,649.9u );
draw (300u ,750u )--(480u ,649.9u );
draw (700u ,750u )--(700u ,649.9u );
draw (900u ,750u )--(900u ,649.9u );
draw (240u ,649.9u )--(240u ,549.9u );
draw (480u ,649.9u )--(480u ,549.9u );
path pp;
pp = (520u ,950u ) .. (500u ,970u ) .. (480u ,950u ) .. (500u ,930u ) .. (520u ,950u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (520u ,950u ) .. (500u ,970u ) .. (480u ,950u ) .. (500u ,930u ) .. (520u ,950u ) .. cycle;
draw pp;
label(btex {\scriptsize 2} etex, (500u ,950u ));
path pp;
pp = (520u ,850u ) .. (500u ,870u ) .. (480u ,850u ) .. (500u ,830u ) .. (520u ,850u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (520u ,850u ) .. (500u ,870u ) .. (480u ,850u ) .. (500u ,830u ) .. (520u ,850u ) .. cycle;
draw pp;
label(btex {\scriptsize 3} etex, (500u ,850u ));
path pp;
pp = (320u ,750u ) .. (300u ,770u ) .. (280u ,750u ) .. (300u ,730u ) .. (320u ,750u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (320u ,750u ) .. (300u ,770u ) .. (280u ,750u ) .. (300u ,730u ) .. (320u ,750u ) .. cycle;
draw pp;
label(btex {\scriptsize 9} etex, (300u ,750u ));
path pp;
pp = (720u ,750u ) .. (700u ,770u ) .. (680u ,750u ) .. (700u ,730u ) .. (720u ,750u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (720u ,750u ) .. (700u ,770u ) .. (680u ,750u ) .. (700u ,730u ) .. (720u ,750u ) .. cycle;
draw pp;
label(btex {\scriptsize 12} etex, (700u ,750u ));
path pp;
pp = (920u ,750u ) .. (900u ,770u ) .. (880u ,750u ) .. (900u ,730u ) .. (920u ,750u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (920u ,750u ) .. (900u ,770u ) .. (880u ,750u ) .. (900u ,730u ) .. (920u ,750u ) .. cycle;
draw pp;
label(btex {\scriptsize 10} etex, (900u ,750u ));
path pp;
pp = (80u ,649.9u ) .. (60u ,669.9u ) .. (40u ,649.9u ) .. (60u ,629.9u ) .. (80u ,649.9u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (80u ,649.9u ) .. (60u ,669.9u ) .. (40u ,649.9u ) .. (60u ,629.9u ) .. (80u ,649.9u ) .. cycle;
draw pp;
label(btex {\scriptsize 6} etex, (60u ,649.9u ));
path pp;
pp = (260u ,649.9u ) .. (240u ,669.9u ) .. (220u ,649.9u ) .. (240u ,629.9u ) .. (260u ,649.9u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (260u ,649.9u ) .. (240u ,669.9u ) .. (220u ,649.9u ) .. (240u ,629.9u ) .. (260u ,649.9u ) .. cycle;
draw pp;
label(btex {\scriptsize 13} etex, (240u ,649.9u ));
path pp;
pp = (500u ,649.9u ) .. (480u ,669.9u ) .. (460u ,649.9u ) .. (480u ,629.9u ) .. (500u ,649.9u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (500u ,649.9u ) .. (480u ,669.9u ) .. (460u ,649.9u ) .. (480u ,629.9u ) .. (500u ,649.9u ) .. cycle;
draw pp;
label(btex {\scriptsize 11} etex, (480u ,649.9u ));
path pp;
pp = (720u ,649.9u ) .. (700u ,669.9u ) .. (680u ,649.9u ) .. (700u ,629.9u ) .. (720u ,649.9u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (720u ,649.9u ) .. (700u ,669.9u ) .. (680u ,649.9u ) .. (700u ,629.9u ) .. (720u ,649.9u ) .. cycle;
draw pp;
label(btex {\scriptsize 14} etex, (700u ,649.9u ));
path pp;
pp = (920u ,649.9u ) .. (900u ,669.9u ) .. (880u ,649.9u ) .. (900u ,629.9u ) .. (920u ,649.9u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (920u ,649.9u ) .. (900u ,669.9u ) .. (880u ,649.9u ) .. (900u ,629.9u ) .. (920u ,649.9u ) .. cycle;
draw pp;
label(btex {\scriptsize 7} etex, (900u ,649.9u ));
path pp;
pp = (260u ,549.9u ) .. (240u ,569.9u ) .. (220u ,549.9u ) .. (240u ,529.9u ) .. (260u ,549.9u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (260u ,549.9u ) .. (240u ,569.9u ) .. (220u ,549.9u ) .. (240u ,529.9u ) .. (260u ,549.9u ) .. cycle;
draw pp;
label(btex {\scriptsize 4} etex, (240u ,549.9u ));
path pp;
pp = (500u ,549.9u ) .. (480u ,569.9u ) .. (460u ,549.9u ) .. (480u ,529.9u ) .. (500u ,549.9u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (500u ,549.9u ) .. (480u ,569.9u ) .. (460u ,549.9u ) .. (480u ,529.9u ) .. (500u ,549.9u ) .. cycle;
draw pp;
label(btex {\scriptsize 8} etex, (480u ,549.9u ));
endfig;

end

