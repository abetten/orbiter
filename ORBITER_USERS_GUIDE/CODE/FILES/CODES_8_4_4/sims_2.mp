% file: sims_2.mp
% created by Orbiter metapost interface
% creation date: Mon Nov 25 12:41:01 MST 2019

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
draw (150u ,285u )--(150u ,255u );
draw (150u ,255u )--(90u ,225u );
draw (150u ,255u )--(225u ,225u );
draw (150u ,255u )--(285u ,225u );
draw (90u ,225u )--(54u ,194.97u );
draw (90u ,225u )--(126u ,194.97u );
draw (90u ,225u )--(162u ,194.97u );
draw (225u ,225u )--(202.5u ,194.97u );
draw (225u ,225u )--(247.5u ,194.97u );
draw (54u ,194.97u )--(27u ,164.97u );
draw (54u ,194.97u )--(81u ,164.97u );
path pp;
pp = (159u ,285u ) .. (150u ,294u ) .. (141u ,285u ) .. (150u ,276u ) .. (159u ,285u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (159u ,285u ) .. (150u ,294u ) .. (141u ,285u ) .. (150u ,276u ) .. (159u ,285u ) .. cycle;
draw pp;
label(btex {\scriptsize 2} etex, (150u ,285u ));
path pp;
pp = (159u ,255u ) .. (150u ,264u ) .. (141u ,255u ) .. (150u ,246u ) .. (159u ,255u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (159u ,255u ) .. (150u ,264u ) .. (141u ,255u ) .. (150u ,246u ) .. (159u ,255u ) .. cycle;
draw pp;
label(btex {\scriptsize 3} etex, (150u ,255u ));
path pp;
pp = (99u ,225u ) .. (90u ,234u ) .. (81u ,225u ) .. (90u ,216u ) .. (99u ,225u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (99u ,225u ) .. (90u ,234u ) .. (81u ,225u ) .. (90u ,216u ) .. (99u ,225u ) .. cycle;
draw pp;
label(btex {\scriptsize 9} etex, (90u ,225u ));
path pp;
pp = (234u ,225u ) .. (225u ,234u ) .. (216u ,225u ) .. (225u ,216u ) .. (234u ,225u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (234u ,225u ) .. (225u ,234u ) .. (216u ,225u ) .. (225u ,216u ) .. (234u ,225u ) .. cycle;
draw pp;
label(btex {\scriptsize 10} etex, (225u ,225u ));
path pp;
pp = (294u ,225u ) .. (285u ,234u ) .. (276u ,225u ) .. (285u ,216u ) .. (294u ,225u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (294u ,225u ) .. (285u ,234u ) .. (276u ,225u ) .. (285u ,216u ) .. (294u ,225u ) .. cycle;
draw pp;
label(btex {\scriptsize 12} etex, (285u ,225u ));
path pp;
pp = (63u ,194.97u ) .. (54u ,203.97u ) .. (45u ,194.97u ) .. (54u ,185.97u ) .. (63u ,194.97u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (63u ,194.97u ) .. (54u ,203.97u ) .. (45u ,194.97u ) .. (54u ,185.97u ) .. (63u ,194.97u ) .. cycle;
draw pp;
label(btex {\scriptsize 11} etex, (54u ,194.97u ));
path pp;
pp = (135u ,194.97u ) .. (126u ,203.97u ) .. (117u ,194.97u ) .. (126u ,185.97u ) .. (135u ,194.97u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (135u ,194.97u ) .. (126u ,203.97u ) .. (117u ,194.97u ) .. (126u ,185.97u ) .. (135u ,194.97u ) .. cycle;
draw pp;
label(btex {\scriptsize 13} etex, (126u ,194.97u ));
path pp;
pp = (171u ,194.97u ) .. (162u ,203.97u ) .. (153u ,194.97u ) .. (162u ,185.97u ) .. (171u ,194.97u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (171u ,194.97u ) .. (162u ,203.97u ) .. (153u ,194.97u ) .. (162u ,185.97u ) .. (171u ,194.97u ) .. cycle;
draw pp;
label(btex {\scriptsize 6} etex, (162u ,194.97u ));
path pp;
pp = (211.5u ,194.97u ) .. (202.5u ,203.97u ) .. (193.5u ,194.97u ) .. (202.5u ,185.97u ) .. (211.5u ,194.97u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (211.5u ,194.97u ) .. (202.5u ,203.97u ) .. (193.5u ,194.97u ) .. (202.5u ,185.97u ) .. (211.5u ,194.97u ) .. cycle;
draw pp;
label(btex {\scriptsize 14} etex, (202.5u ,194.97u ));
path pp;
pp = (256.5u ,194.97u ) .. (247.5u ,203.97u ) .. (238.5u ,194.97u ) .. (247.5u ,185.97u ) .. (256.5u ,194.97u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (256.5u ,194.97u ) .. (247.5u ,203.97u ) .. (238.5u ,194.97u ) .. (247.5u ,185.97u ) .. (256.5u ,194.97u ) .. cycle;
draw pp;
label(btex {\scriptsize 7} etex, (247.5u ,194.97u ));
path pp;
pp = (36u ,164.97u ) .. (27u ,173.97u ) .. (18u ,164.97u ) .. (27u ,155.97u ) .. (36u ,164.97u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (36u ,164.97u ) .. (27u ,173.97u ) .. (18u ,164.97u ) .. (27u ,155.97u ) .. (36u ,164.97u ) .. cycle;
draw pp;
label(btex {\scriptsize 4} etex, (27u ,164.97u ));
path pp;
pp = (90u ,164.97u ) .. (81u ,173.97u ) .. (72u ,164.97u ) .. (81u ,155.97u ) .. (90u ,164.97u ) .. cycle;
fill pp withcolor 1 white;
path pp;
pp = (90u ,164.97u ) .. (81u ,173.97u ) .. (72u ,164.97u ) .. (81u ,155.97u ) .. (90u ,164.97u ) .. cycle;
draw pp;
label(btex {\scriptsize 8} etex, (81u ,164.97u ));
endfig;

end

