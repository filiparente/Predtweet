import read_embbedings as run
import itertools
import subprocess
import progressbar
import os

dt = [3,4,6,12,24,48]  #discretization unit in hours
dw = [1,3,5,7] #length of the sliding window of previous features, in units of dt

all_combs = list(itertools.product(dt, dw))

bar = progressbar.ProgressBar(maxval=len(all_combs), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
n_comb = 0
for comb in all_combs:
    dt = comb[0]
    dw = comb[1]
    bar.update(n_comb+1)
    n_comb+=1
    
    run.main(window_size=dw, disc_unit=dt, out_path="results/")

    # using Popen may suit better here depending on how you want to deal
    # with the output of the child_script.
    #subprocess.call(["python", "read_embbedings.py", "--window_size", str(dw), "--discretization_unit", str(dt), "--out_path", 'results/'], shell=True)
    #os.system("python read_embbedings.py --window_size "+str(dw)+" --discretization_unit "+str(dt)+" --out_path results/")

bar.finish()