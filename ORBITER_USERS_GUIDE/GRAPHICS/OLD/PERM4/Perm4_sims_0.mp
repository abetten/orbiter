% file: Perm4_sims_0.mp
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
draw (500u ,937.5u )--(500u ,812.5u );
draw (500u ,812.5u )--(500u ,687.5u );
draw (500u ,687.5u )--(500u ,562.5u );
path pp;
pp = (520u ,937.5u ) .. (500u ,957.5u ) .. (480u ,937.5u ) .. (500u ,917.5u ) .. (520u ,937.5u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (520u ,937.5u ) .. (500u ,957.5u ) .. (480u ,937.5u ) .. (500u ,917.5u ) .. (520u ,937.5u ) .. cycle;
draw pp;
label(btex {\scriptsize 0} etex, (500u ,937.5u ));
path pp;
pp = (520u ,812.5u ) .. (500u ,832.5u ) .. (480u ,812.5u ) .. (500u ,792.5u ) .. (520u ,812.5u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (520u ,812.5u ) .. (500u ,832.5u ) .. (480u ,812.5u ) .. (500u ,792.5u ) .. (520u ,812.5u ) .. cycle;
draw pp;
label(btex {\scriptsize 1} etex, (500u ,812.5u ));
path pp;
pp = (520u ,687.5u ) .. (500u ,707.5u ) .. (480u ,687.5u ) .. (500u ,667.5u ) .. (520u ,687.5u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (520u ,687.5u ) .. (500u ,707.5u ) .. (480u ,687.5u ) .. (500u ,667.5u ) .. (520u ,687.5u ) .. cycle;
draw pp;
label(btex {\scriptsize 2} etex, (500u ,687.5u ));
path pp;
pp = (520u ,562.5u ) .. (500u ,582.5u ) .. (480u ,562.5u ) .. (500u ,542.5u ) .. (520u ,562.5u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (520u ,562.5u ) .. (500u ,582.5u ) .. (480u ,562.5u ) .. (500u ,542.5u ) .. (520u ,562.5u ) .. cycle;
draw pp;
label(btex {\scriptsize 3} etex, (500u ,562.5u ));
endfig;

end

