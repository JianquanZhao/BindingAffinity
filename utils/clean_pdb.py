import os
from parse_args import get_args

if __name__ == '__main__':
    args = get_args()
    files= os.listdir(args.inputdir)
    flag=False
    for file in files:
        if not os.path.isdir(file): 
            infp=args.inputdir+'/'+file
            outfp=args.outdir+'/'+file.split('.')[0]+'.pdb'
            thr_flag = False
            with open(infp, "r") as inputFile,open(outfp,"w") as outFile:
                for line in inputFile:
                    if line.startswith('MODEL') and line[12:14]!=' 1':
                        flag=True
                        continue
                    elif line.startswith("ENDMDL"): 
                        flag=False
                        continue
                    if  (line.startswith("ATOM") or line.startswith("TER") ) and flag==False:
                        if line.startswith("TER"):
                            if thr_flag == True:
                                continue
                            thr_flag = True
                        if line.startswith("ATOM") and line[17:20] == 'UNK':
                            continue
                        if line.startswith("HETATM") and line[17:20] == 'HOH':
                            continue
                        
                        if line.startswith("ATOM"):
                            thr_flag = False
                        outFile.write(line)