[ChemRxiv](https://chemrxiv.org/engage/chemrxiv/article-details/6221f17357a9d20c9a729ecb)  |  [Paper](https://pubs.rsc.org/en/content/articlelanding/2023/dd/d2dd00086e)

# Install

To run CVAE you will need some packages, which are dependent on your computers specifications. 
This includes [PyTorch](https://pytorch.org/).

All other packages can be install through the requirement files or the install.sh file. 
First go to your desired conda environment.
 ```
conda activate <env_name>
``` 
Or create one and activate it afterwards.
```
conda create --name env_name python=3.7
``` 
Now install the required packages through the requirement files and then install DiffPy-CMI (see how to [HERE](https://www.diffpy.org/products/diffpycmi/index.html)).
```
pip install -r requirements_pip.txt
conda install --file requirements_conda.txt
``` 
Or install through the install.sh file.
```
sh install.sh
``` 

