% file: PGGL_2_8_OnConic_sims_2.mp
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
draw (500u ,950u )--(500u ,850u );
draw (500u ,850u )--(500u ,750u );
draw (500u ,750u )--(250u ,649.9u );
draw (500u ,750u )--(750u ,649.9u );
draw (250u ,649.9u )--(250u ,549.9u );
draw (750u ,649.9u )--(750u ,549.9u );
path pp;
pp = (525u ,950u ) .. (500u ,975u ) .. (475u ,950u ) .. (500u ,925u ) .. (525u ,950u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (525u ,950u ) .. (500u ,975u ) .. (475u ,950u ) .. (500u ,925u ) .. (525u ,950u ) .. cycle;
draw pp;
label(btex {\scriptsize 2} etex, (500u ,950u ));
path pp;
pp = (525u ,850u ) .. (500u ,875u ) .. (475u ,850u ) .. (500u ,825u ) .. (525u ,850u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (525u ,850u ) .. (500u ,875u ) .. (475u ,850u ) .. (500u ,825u ) .. (525u ,850u ) .. cycle;
draw pp;
label(btex {\scriptsize 3} etex, (500u ,850u ));
path pp;
pp = (525u ,750u ) .. (500u ,775u ) .. (475u ,750u ) .. (500u ,725u ) .. (525u ,750u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (525u ,750u ) .. (500u ,775u ) .. (475u ,750u ) .. (500u ,725u ) .. (525u ,750u ) .. cycle;
draw pp;
label(btex {\scriptsize 5} etex, (500u ,750u ));
path pp;
pp = (275u ,649.9u ) .. (250u ,674.9u ) .. (225u ,649.9u ) .. (250u ,624.9u ) .. (275u ,649.9u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (275u ,649.9u ) .. (250u ,674.9u ) .. (225u ,649.9u ) .. (250u ,624.9u ) .. (275u ,649.9u ) .. cycle;
draw pp;
label(btex {\scriptsize 8} etex, (250u ,649.9u ));
path pp;
pp = (775u ,649.9u ) .. (750u ,674.9u ) .. (725u ,649.9u ) .. (750u ,624.9u ) .. (775u ,649.9u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (775u ,649.9u ) .. (750u ,674.9u ) .. (725u ,649.9u ) .. (750u ,624.9u ) .. (775u ,649.9u ) .. cycle;
draw pp;
label(btex {\scriptsize 6} etex, (750u ,649.9u ));
path pp;
pp = (275u ,549.9u ) .. (250u ,574.9u ) .. (225u ,549.9u ) .. (250u ,524.9u ) .. (275u ,549.9u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (275u ,549.9u ) .. (250u ,574.9u ) .. (225u ,549.9u ) .. (250u ,524.9u ) .. (275u ,549.9u ) .. cycle;
draw pp;
label(btex {\scriptsize 4} etex, (250u ,549.9u ));
path pp;
pp = (775u ,549.9u ) .. (750u ,574.9u ) .. (725u ,549.9u ) .. (750u ,524.9u ) .. (775u ,549.9u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (775u ,549.9u ) .. (750u ,574.9u ) .. (725u ,549.9u ) .. (750u ,524.9u ) .. (775u ,549.9u ) .. cycle;
draw pp;
label(btex {\scriptsize 7} etex, (750u ,549.9u ));
endfig;

end

