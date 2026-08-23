% file: PGL_2_3_group_table_order_6.mp
% created by Orbiter metapost interface
% creation date: Fri Nov  8 20:43:55 +03 2024

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
% box outline
draw (0u ,99.995u )--(0u ,-899.964u );
draw (0u ,-899.964u )--(999.96u ,-899.964u );
draw (999.96u ,-899.964u )--(999.96u ,99.995u );
draw (999.96u ,99.995u )--(0u ,99.995u );
pickup pencircle scaled 0.1pt;
label(btex 0 etex, (-166.66u ,16.666u ));
label(btex 1 etex, (-166.66u ,-149.994u ));
label(btex 2 etex, (-166.66u ,-316.654u ));
label(btex 3 etex, (-166.66u ,-483.314u ));
label(btex 4 etex, (-166.66u ,-649.974u ));
label(btex 5 etex, (-166.66u ,-816.634u ));
label(btex 0 etex, (83.33u ,266.656u ));
label(btex 1 etex, (249.99u ,266.656u ));
label(btex 2 etex, (416.65u ,266.656u ));
label(btex 3 etex, (583.31u ,266.656u ));
label(btex 4 etex, (749.97u ,266.656u ));
label(btex 5 etex, (916.63u ,266.656u ));
pickup pencircle scaled 0.1pt;
% 0_1
fill buildcycle((166.66u ,99.995u )--(166.66u ,-66.664u )--(333.32u ,-66.664u )--(333.32u ,99.995u )--(166.66u ,99.995u ) -- cycle ) withcolor 1 black;
% 0_2
fill buildcycle((333.32u ,99.995u )--(333.32u ,-66.664u )--(499.98u ,-66.664u )--(499.98u ,99.995u )--(333.32u ,99.995u ) -- cycle ) withcolor 1% 0_3
fill buildcycle((499.98u ,99.995u )--(499.98u ,-66.664u )--(666.64u ,-66.664u )--(666.64u ,99.995u )--(499.98u ,99.995u ) -- cycle ) withcolor 1% 0_4
fill buildcycle((666.64u ,99.995u )--(666.64u ,-66.664u )--(833.3u ,-66.664u )--(833.3u ,99.995u )--(666.64u ,99.995u ) -- cycle ) withcolor 1% 0_5
fill buildcycle((833.3u ,99.995u )--(833.3u ,-66.664u )--(999.96u ,-66.664u )--(999.96u ,99.995u )--(833.3u ,99.995u ) -- cycle ) withcolor 1% 1_0
fill buildcycle((0u ,-66.664u )--(0u ,-233.324u )--(166.66u ,-233.324u )--(166.66u ,-66.664u )--(0u ,-66.664u ) -- cycle ) withcolor 1 black;
% 1_2
fill buildcycle((333.32u ,-66.664u )--(333.32u ,-233.324u )--(499.98u ,-233.324u )--(499.98u ,-66.664u )--(333.32u ,-66.664u ) -- cycle ) withcolor 1% 1_3
fill buildcycle((499.98u ,-66.664u )--(499.98u ,-233.324u )--(666.64u ,-233.324u )--(666.64u ,-66.664u )--(499.98u ,-66.664u ) -- cycle ) withcolor 1% 1_4
fill buildcycle((666.64u ,-66.664u )--(666.64u ,-233.324u )--(833.3u ,-233.324u )--(833.3u ,-66.664u )--(666.64u ,-66.664u ) -- cycle ) withcolor 1% 1_5
fill buildcycle((833.3u ,-66.664u )--(833.3u ,-233.324u )--(999.96u ,-233.324u )--(999.96u ,-66.664u )--(833.3u ,-66.664u ) -- cycle ) withcolor 1% 2_0
fill buildcycle((0u ,-233.324u )--(0u ,-399.983u )--(166.66u ,-399.983u )--(166.66u ,-233.324u )--(0u ,-233.324u ) -- cycle ) withcolor 1% 2_1
fill buildcycle((166.66u ,-233.324u )--(166.66u ,-399.983u )--(333.32u ,-399.983u )--(333.32u ,-233.324u )--(166.66u ,-233.324u ) -- cycle ) withcolor 1% 2_2
fill buildcycle((333.32u ,-233.324u )--(333.32u ,-399.983u )--(499.98u ,-399.983u )--(499.98u ,-233.324u )--(333.32u ,-233.324u ) -- cycle ) withcolor 1% 2_3
fill buildcycle((499.98u ,-233.324u )--(499.98u ,-399.983u )--(666.64u ,-399.983u )--(666.64u ,-233.324u )--(499.98u ,-233.324u ) -- cycle ) withcolor 1 black;
% 2_4
fill buildcycle((666.64u ,-233.324u )--(666.64u ,-399.983u )--(833.3u ,-399.983u )--(833.3u ,-233.324u )--(666.64u ,-233.324u ) -- cycle ) withcolor 1% 3_0
fill buildcycle((0u ,-399.983u )--(0u ,-566.644u )--(166.66u ,-566.644u )--(166.66u ,-399.983u )--(0u ,-399.983u ) -- cycle ) withcolor 1% 3_1
fill buildcycle((166.66u ,-399.983u )--(166.66u ,-566.644u )--(333.32u ,-566.644u )--(333.32u ,-399.983u )--(166.66u ,-399.983u ) -- cycle ) withcolor 1% 3_2
fill buildcycle((333.32u ,-399.983u )--(333.32u ,-566.644u )--(499.98u ,-566.644u )--(499.98u ,-399.983u )--(333.32u ,-399.983u ) -- cycle ) withcolor 1% 3_4
fill buildcycle((666.64u ,-399.983u )--(666.64u ,-566.644u )--(833.3u ,-566.644u )--(833.3u ,-399.983u )--(666.64u ,-399.983u ) -- cycle ) withcolor 1% 3_5
fill buildcycle((833.3u ,-399.983u )--(833.3u ,-566.644u )--(999.96u ,-566.644u )--(999.96u ,-399.983u )--(833.3u ,-399.983u ) -- cycle ) withcolor 1 black;
% 4_0
fill buildcycle((0u ,-566.644u )--(0u ,-733.304u )--(166.66u ,-733.304u )--(166.66u ,-566.644u )--(0u ,-566.644u ) -- cycle ) withcolor 1% 4_1
fill buildcycle((166.66u ,-566.644u )--(166.66u ,-733.304u )--(333.32u ,-733.304u )--(333.32u ,-566.644u )--(166.66u ,-566.644u ) -- cycle ) withcolor 1% 4_2
fill buildcycle((333.32u ,-566.644u )--(333.32u ,-733.304u )--(499.98u ,-733.304u )--(499.98u ,-566.644u )--(333.32u ,-566.644u ) -- cycle ) withcolor 1 black;
% 4_3
fill buildcycle((499.98u ,-566.644u )--(499.98u ,-733.304u )--(666.64u ,-733.304u )--(666.64u ,-566.644u )--(499.98u ,-566.644u ) -- cycle ) withcolor 1% 4_5
fill buildcycle((833.3u ,-566.644u )--(833.3u ,-733.304u )--(999.96u ,-733.304u )--(999.96u ,-566.644u )--(833.3u ,-566.644u ) -- cycle ) withcolor 1% 5_0
fill buildcycle((0u ,-733.304u )--(0u ,-899.964u )--(166.66u ,-899.964u )--(166.66u ,-733.304u )--(0u ,-733.304u ) -- cycle ) withcolor 1% 5_1
fill buildcycle((166.66u ,-733.304u )--(166.66u ,-899.964u )--(333.32u ,-899.964u )--(333.32u ,-733.304u )--(166.66u ,-733.304u ) -- cycle ) withcolor 1% 5_3
fill buildcycle((499.98u ,-733.304u )--(499.98u ,-899.964u )--(666.64u ,-899.964u )--(666.64u ,-733.304u )--(499.98u ,-733.304u ) -- cycle ) withcolor 1% 5_4
fill buildcycle((666.64u ,-733.304u )--(666.64u ,-899.964u )--(833.3u ,-899.964u )--(833.3u ,-733.304u )--(666.64u ,-733.304u ) -- cycle ) withcolor 1 black;
% 5_5
fill buildcycle((833.3u ,-733.304u )--(833.3u ,-899.964u )--(999.96u ,-899.964u )--(999.96u ,-733.304u )--(833.3u ,-733.304u ) -- cycle ) withcolor 1endfig;

end

