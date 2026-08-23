% file: PGO_5_2_sims_2.mp
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
draw (500u ,875u )--(250u ,625u );
draw (500u ,875u )--(750u ,625u );
path pp;
pp = (520u ,875u ) .. (500u ,895u ) .. (480u ,875u ) .. (500u ,855u ) .. (520u ,875u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (520u ,875u ) .. (500u ,895u ) .. (480u ,875u ) .. (500u ,855u ) .. (520u ,875u ) .. cycle;
draw pp;
label(btex {\scriptsize 2} etex, (500u ,875u ));
path pp;
pp = (270u ,625u ) .. (250u ,645u ) .. (230u ,625u ) .. (250u ,605u ) .. (270u ,625u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (270u ,625u ) .. (250u ,645u ) .. (230u ,625u ) .. (250u ,605u ) .. (270u ,625u ) .. cycle;
draw pp;
label(btex {\scriptsize 12} etex, (250u ,625u ));
path pp;
pp = (770u ,625u ) .. (750u ,645u ) .. (730u ,625u ) .. (750u ,605u ) .. (770u ,625u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (770u ,625u ) .. (750u ,645u ) .. (730u ,625u ) .. (750u ,605u ) .. (770u ,625u ) .. cycle;
draw pp;
label(btex {\scriptsize 5} etex, (750u ,625u ));
endfig;

end

