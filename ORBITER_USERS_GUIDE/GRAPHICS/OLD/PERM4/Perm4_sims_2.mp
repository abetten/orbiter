% file: Perm4_sims_2.mp
% created by Orbiter metapost interface
% creation date: Sat Dec  4 21:21:38 +03 2021

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
label(btex {\scriptsize 2} etex, (500u ,875u ));
path pp;
pp = (520u ,625u ) .. (500u ,645u ) .. (480u ,625u ) .. (500u ,605u ) .. (520u ,625u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (520u ,625u ) .. (500u ,645u ) .. (480u ,625u ) .. (500u ,605u ) .. (520u ,625u ) .. cycle;
draw pp;
label(btex {\scriptsize 3} etex, (500u ,625u ));
endfig;

end

