ABOUT
-------------------------------------------------------------------------------
FWQPBO is a Python command-line tool for MRI [chemical shift based fat-water separation with B0-correction based on QPBO graph cuts](https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.26479). 

Input paramaters are provided by human readable configuration files. 

Input data has to be in DICOM format (standard or enhanced).

INSTALLATION
-------------------------------------------------------------------------------
Install dependencies by running `pip install -r requirements.txt` from a terminal located in the fwqpbo folder.

HOW TO USE
-------------------------------------------------------------------------------
Once the required packages are installed, update the parameter files as needed and run the following command:
```
./main.py -d params/data.yml -a params/algo_3D.yml -m params/model.yml
```

Check [HOWTO.md](HOWTO.md) for more info about parameter files.

FORK UPDATES
-------------------------------------------------------------------------------
The main differences between this project and the original one (available at [github/bertglun/fwqpbo](https://github.com/bretglun/fwqpbo)) are:
- added support of Enhanced DICOM files
- PEP8 formatting + on its way to have sphinx compatible comments
- added compatibility with latest pydicom version (3.0.1)
- removed MatLab compatibility and obsolete ISMRM 2012 data (no longer available)


HOW TO CITE
-------------------------------------------------------------------------------
Berglund J and Skorpil M. *Multi-scale graph-cut algorithm for efficient water-
fat separation*. Magn Reson Med, 78(3):941-949, 2017. [doi: 10.1002/mrm.26479]



CONTACT INFORMATION
-------------------------------------------------------------------------------

For any question regarding the method:
Johan Berglund, Ph.D.  
Dept. of Clinical Neuroscience  
Karolinska Institutet,  
Stockholm, Sweden  
johan.berglund@ki.se  

For any bug or question related to this version of the code, feel free to [open an issue](https://github.com/eliopato/fwqpbo/issues)

LICENCE
-------------------------------------------------------------------------------

*Copyright (c) 2016–2020 Johan Berglund*

*FWQPBO is distributed under the terms of the GNU General Public License*

*This program is free software: you can redistribute it and/or modify*
*it under the terms of the GNU General Public License as published by*
*the Free Software Foundation, either version 3 of the License, or*
*(at your option) any later version.*

*This program is distributed in the hope that it will be useful,*
*but WITHOUT ANY WARRANTY; without even the implied warranty of*
*MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the*
*GNU General Public License for more details.*

*You should have received a copy of the GNU General Public License*
*along with this program.  If not, see <http://www.gnu.org/licenses/>.*