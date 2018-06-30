import PSO
import anneal
import time
with open('results.txt','a') as out:
        out.write('\n')
        out.write(time.asctime(time.localtime(time.time())))
for i in range(5):
    start = time.time()
    result = PSO.pso_run()
    result_names = [
            'similar params: ',
            'similar fitness: ',
            'hebb + comb params: ',
            'hebb + comb fitness: '
            ]
#    result = anneal.run_anneal()
    with open('results.txt','a') as out:
        for item in range(len(result)):
            out.write('\n')
            out.write(str(result_names[item])+str(list(result)[item]))
        out.write('\n')
        out.write('finalparams: ')
        out.write(str(list(result[0][:-9])+list(result[2])))
        out.write('\n')

    with open('exper.txt','a') as out2:
        out2.write('\n')
        out2.write(str(result[3]))
        out2.write('\n')
        out2.write(str(time.time()-start))
        out2.write('\n')