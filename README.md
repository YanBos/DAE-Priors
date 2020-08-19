# Leveraging Deep Denoising Autoencoders for Disparity Estimation and Restoration

In this work the source code for the Bachelors project is included. It contains all code necessary to construct the models used and also some of the trained networks analysed in the documentation. 

Please use <code>pip3 install -r requirements.txt</code> in order to install the required dependencies. The code can be run with python 3.7.

New networks can theoretically be trained via the files <code>train_\*.py</code>. Training data must be provided in the <code>training_data</code> directory. The aforementioned files can be run from the command line by making them executable via <code>chmod +x ABS_PATH/train_\*.py</code> (Please view <code>parser/parser.py</code> for the command line options).

All other files as mentioned can be executed if the data in the <code>data</code> data directory is provided. It can be found at <cite>[Lightfield analysis][3]</cite>. Putting the corresponding lightfields into the <code>data</code> folder will enable you to execute the remaining python files.

The code for the general scheme by <cite>[Zach et al.][1]</cite>[1] was translated to python from the code of <cite>[Sanchez et al.][2]</cite>[2]. The code presented also contains the scripts used for the optimization of disparity maps for inpainting, deblurring and super resolution.


[1] Zach, Christopher & Pock, Thomas & Bischof, Horst. (2007). A Duality Based Approach for Realtime TV-L1 Optical Flow. Pattern Recognition. 4713. 214-223. 10.1007/978-3-540-74936-3_22. 

[2] Sánchez Pérez, Javier and Meinhardt-Llopis, Enric and Facciolo, Gabriele, TV-L1 Optical Flow Estimation, Image Processing On Line, Volume 3, pp. 137-150, 2013.

[1]: https://www.researchgate.net/publication/248964741_A_Duality_Based_Approach_for_Realtime_TV-L1_Optical_Flow
[2]: http://www.ipol.im/pub/art/2013/26/
[3]: https://lightfield-analysis.uni-konstanz.de 
[4]: https://cloud.uni-konstanz.de/index.php/s/HMQWdcGkK6XZjWG