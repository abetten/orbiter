% file: sims_6.mp
% created by Orbiter metapost interface
% creation date: Mon Nov 25 02:25:56 MST 2019

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
path pp;
pp = (159u ,225u ) .. (150u ,234u ) .. (141u ,225u ) .. (150u ,216u ) .. (159u ,225u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (159u ,225u ) .. (150u ,234u ) .. (141u ,225u ) .. (150u ,216u ) .. (159u ,225u ) .. cycle;
draw pp;
label(btex {\scriptsize 6} etex, (150u ,225u ));
endfig;

end

