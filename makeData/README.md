[arXiv]XXX  |  [Paper] XXX

# Data introduction
  

1. [Generate data](#generate-data)
    1. [Generate data arguments](#generate-data-arguments)
2. [Generel data structure](#generel-data-structure)
    1. [Mono-Metallic Nanoparticles (MMNPs)](#mono-metallic-nanoparticles-mmnps)
    2. [Graph representation](#graph-representation)
    3. [Pair Distribution Function (PDF)](#pair-distribution-function-pdf)

# Generate XYZ files
[DiffPy-CMI](https://www.diffpy.org/products/diffpycmi/index.html) in required to simulate PDFs, which only runs on Linux or macOS. To run it on a Windows computer
please use the [Ubuntu subsystem](https://ubuntu.com/tutorials/ubuntu-on-windows#1-overview). To generate more data run the simPDF_xyz.py script. 

# Generate graph-inspired tabular data 
To generate graph-inspired tabular data run the graph_maker.py script. 
  
# Generel data structure
A simplified description is shown below. For detailed description of the data format please revisit the paper.

## Mono-Metallic Nanoparticles (MMNPs)
The MMNPs are described using a XYZ format describing the element and their euclidian distances as seen below:

Atom<sub>1</sub> &nbsp; &nbsp; x<sub>1</sub> &nbsp; &nbsp; y<sub>1</sub> &nbsp; &nbsp; z<sub>1</sub> <br>
Atom<sub>2</sub> &nbsp; &nbsp; x<sub>2</sub> &nbsp; &nbsp; y<sub>2</sub> &nbsp; &nbsp; z<sub>2</sub> <br>
...  
Atom<sub>N</sub> &nbsp; &nbsp; x<sub>N</sub> &nbsp; &nbsp; y<sub>N</sub> &nbsp; &nbsp; z<sub>N</sub> <br>

## Graph representation
Each structure in graph representation can be described as, G = (X,A), where X ∈ R<sup>N×F</sup> is the node feature matrix which contains F features that can describe each of the N atoms in the structure. We use F = 3 comprising only the Euclidean coordinates of the atom in a 3-dimensional space. The interatomic relationships are captured using the adjacency matrix A ∈ R<sup>N×N</sup>. In our case, the entries of the adjacency matrix are the Euclidean distance between pairs of atoms resulting in a soft adjacency matrix. However, when the distance between any pair of nodes is larger than the lattice constant the corresponding edge weight is set to zero. 

This CVAE does not use graphs as input but simply concatenate the adjacency matrix to the XYZ coordinates as input to the CVAE. For a graph-based CVAE visit  [DeepStruc](https://github.com/EmilSkaaning/DeepStruc.git).

## Pair Distribution Function (PDF)
The PDF is the Fourier transform of total scattering data, which can be obtained through x-ray, neutron or electron scattering.
G(r) can be interpreted as a histogram of real-space interatomic distances and the information is equivalent to that of an unassigned distance matrix. <br> 
A simulated PDF and how we normalise them are shown below:
![alt text](../img/PDF.png "Simulated PDF")

