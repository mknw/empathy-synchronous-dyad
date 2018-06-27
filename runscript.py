import PSO
import time
for i in range(5):
    start = time.time()
    result = PSO.pso_run()
    with open('results.txt','a') as out:
        out.write('\n\n')
        out.write(str(result))
    with open('exper.txt','a') as out2:
        out2.write('\n\n')
        out2.write(str(result[1]))
        out2.write(str(time.time()-start))
