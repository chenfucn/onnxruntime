# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------


import os
import numpy as np
import numpy.typing as npt
import subprocess
import tempfile

class Q4dqWrapper:
    """ A wrapper to native command line onnxruntime_mlas_q4dq 
    """

    def __init__(
        self,
        exepath
    ):
        self.q4dq_cmd = exepath

    def quantize(self, fp32weight: npt.ArrayLike) -> np.ndarray:
        """4b quantize fp32 weight to a blob"""

        array = fp32weight.astype(np.float32)
        if len(array.shape) != 2:
            raise Exception('Only 2D fp32 array accepted!')
        rows, cols = array.shape
        
        with tempfile.TemporaryDirectory() as tmpdirname:
            fp32file = os.path.join(tmpdirname, 'fp32weight')
            print(' Writing fp32 weight to file: ', tmpdirname)
            array.tofile(fp32file)

            q4file = os.path.join(tmpdirname, 'q4weight')

            cmd = '{cmdpath} q {k} {n} --input_file {fp32} --output_file {q4} --output_format bin'.format(
                cmdpath=self.q4dq_cmd, k=rows, n=cols, fp32=fp32file, q4=q4file)
            subprocess.run(cmd, shell=True)

            packed = np.fromfile(q4file, dtype='uint8')
            return packed

