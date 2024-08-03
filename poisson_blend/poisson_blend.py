import subprocess
import sys

def poisson_blend(target_file, source_file, mask_file, output_file):
    subprocess.run(['./poisson_blend/build/poisson_blend', '-source', source_file, '-target', target_file, '-mask', mask_file, '-output', output_file])


