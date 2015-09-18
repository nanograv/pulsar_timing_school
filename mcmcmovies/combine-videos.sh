#! /bin/bash
# Combine 4 MCMC videos, as outputted by the mcmcmovie.py script. Make sure the
# sample times for all four of the MCMC runs are the same for fair comparison.

ffmpeg -i $1 -i $2 -i $3 -i $4 -filter_complex "[0:0]pad=iw*2:ih*2[a];[a][1:0]overlay=w[b];[b][2:0]overlay=0:h[c];[c][3:0]overlay=w:h" -shortest combinedmovie.avi
