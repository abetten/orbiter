% file: PGO_5_2_sims_4.mp
% created by Orbiter metapost interface
% creation date: Mon Dec 21 16:40:00 MST 2020

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
pp = (520u ,750u ) .. (500u ,770u ) .. (480u ,750u ) .. (500u ,730u ) .. (520u ,750u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (520u ,750u ) .. (500u ,770u ) .. (480u ,750u ) .. (500u ,730u ) .. (520u ,750u ) .. cycle;
draw pp;
label(btex {\scriptsize 4} etex, (500u ,750u ));
endfig;

end

