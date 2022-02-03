File Edit Options Buffers Tools Insert Help                                                    
#!/bin/sh                                                                                      

#$ -S /bin/sh                                                                                  
#$ -cwd                                                                                        
#$ -V                                                                                          
#$ -q all.q@messier11                                                                                                               
#$ -o ./Log.txt                                         
##$ -e ./Log.txt                                          
#$ -j y                                                                                        
#$ -pe openmpi 24                                                                               

#$ -N Spec_PMF                                                                                  

python ./exec2.py



