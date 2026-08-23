% file: Perm4_sims_1.mp
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
draw (500u ,916.6u )--(500u ,750u );
draw (500u ,750u )--(500u ,583.3u );
path pp;
pp = (520u ,916.6u ) .. (500u ,936.6u ) .. (480u ,916.6u ) .. (500u ,896.6u ) .. (520u ,916.6u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (520u ,916.6u ) .. (500u ,936.6u ) .. (480u ,916.6u ) .. (500u ,896.6u ) .. (520u ,916.6u ) .. cycle;
draw pp;
label(btex {\scriptsize 1} etex, (500u ,916.6u ));
path pp;
pp = (520u ,750u ) .. (500u ,770u ) .. (480u ,750u ) .. (500u ,730u ) .. (520u ,750u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (520u ,750u ) .. (500u ,770u ) .. (480u ,750u ) .. (500u ,730u ) .. (520u ,750u ) .. cycle;
draw pp;
label(btex {\scriptsize 2} etex, (500u ,750u ));
path pp;
pp = (520u ,583.3u ) .. (500u ,603.3u ) .. (480u ,583.3u ) .. (500u ,563.3u ) .. (520u ,583.3u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (520u ,583.3u ) .. (500u ,603.3u ) .. (480u ,583.3u ) .. (500u ,563.3u ) .. (520u ,583.3u ) .. cycle;
draw pp;
label(btex {\scriptsize 3} etex, (500u ,583.3u ));
endfig;

end

