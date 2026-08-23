% file: PGGL_2_8_OnConic_sims_3.mp
% created by Orbiter metapost interface
% creation date: Sat Dec 19 23:54:03 MST 2020

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
pp = (525u ,916.6u ) .. (500u ,941.6u ) .. (475u ,916.6u ) .. (500u ,891.6u ) .. (525u ,916.6u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (525u ,916.6u ) .. (500u ,941.6u ) .. (475u ,916.6u ) .. (500u ,891.6u ) .. (525u ,916.6u ) .. cycle;
draw pp;
label(btex {\scriptsize 4} etex, (500u ,916.6u ));
path pp;
pp = (525u ,750u ) .. (500u ,775u ) .. (475u ,750u ) .. (500u ,725u ) .. (525u ,750u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (525u ,750u ) .. (500u ,775u ) .. (475u ,750u ) .. (500u ,725u ) .. (525u ,750u ) .. cycle;
draw pp;
label(btex {\scriptsize 6} etex, (500u ,750u ));
path pp;
pp = (525u ,583.3u ) .. (500u ,608.3u ) .. (475u ,583.3u ) .. (500u ,558.3u ) .. (525u ,583.3u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (525u ,583.3u ) .. (500u ,608.3u ) .. (475u ,583.3u ) .. (500u ,558.3u ) .. (525u ,583.3u ) .. cycle;
draw pp;
label(btex {\scriptsize 7} etex, (500u ,583.3u ));
endfig;

end

