% file: PGO_5_2_sims_5.mp
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
draw (500u ,875u )--(500u ,625u );
path pp;
pp = (520u ,875u ) .. (500u ,895u ) .. (480u ,875u ) .. (500u ,855u ) .. (520u ,875u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (520u ,875u ) .. (500u ,895u ) .. (480u ,875u ) .. (500u ,855u ) .. (520u ,875u ) .. cycle;
draw pp;
label(btex {\scriptsize 5} etex, (500u ,875u ));
path pp;
pp = (520u ,625u ) .. (500u ,645u ) .. (480u ,625u ) .. (500u ,605u ) .. (520u ,625u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (520u ,625u ) .. (500u ,645u ) .. (480u ,625u ) .. (500u ,605u ) .. (520u ,625u ) .. cycle;
draw pp;
label(btex {\scriptsize 12} etex, (500u ,625u ));
endfig;

end

