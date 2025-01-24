import numpy as np
import cv2 as cv
import tifffile
import argparse
import os

def avg_stacks(img,depth):
    if img.shape[0] % depth != 0:
        print(img.shape[0],depth)
        raise ValueError("Incompatibe stack size with depth")

    n_stacks = int(img.shape[0]/depth)
    stack = np.zeros((depth,img.shape[-2],img.shape[-1]))

    for i in range(n_stacks):
        stack+=img[depth*i:depth*i+depth]/n_stacks

    stack = np.array(stack,np.float32)
    return stack



parser = argparse.ArgumentParser(description='Stack Averager.')
parser.add_argument('--depth', metavar='D', type=int, nargs=1,
                    help='Depth of stack')
parser.add_argument('--path', metavar='P', type=str, nargs=1,
                    help='Path to stack')
parser.add_argument('--save_path', metavar='S', type=str, nargs=1,
                    help='Path to save avg stack')

parser.add_argument('--make_steps', metavar='A', type=bool, nargs=1,
                    help='Path to save avg stack')

args = parser.parse_args()

depth = args.depth[0]
stack = tifffile.imread(args.path[0])
print(stack.shape)
n_stacks = int(stack.shape[0]/depth)
if args.make_steps:
    for i in range(2,n_stacks-1):
        avg_stack = avg_stacks(stack[depth:(i+1)*depth],depth)
        p = os.path.join(args.save_path[0],"{f}x.tif".format(f = i))
        tifffile.imsave(p, avg_stack)
else:

    avg_stack = avg_stacks(stack,depth)
    p = os.path.join(args.save_path[0],"{f}x.tif".format(f = n_stacks))    
    tifffile.imsave(p, avg_stack)
