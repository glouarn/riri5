=========================
README for RIRI5
=========================

This is RIRI5 model, an adaptation of the RIRI model in python for simple sky discretisations (diffuse sky) and multi-species canpies

See 
Sinoquet, H., Le Roux, X., Adam, B., Ameglio, T., & Daudet, F. A. (2001). RATP: a model for simulating the spatial distribution of radiation absorption, transpiration and photosynthesis within canopies: application to an isolated tree crown. Plant, Cell & Environment, 24(4), 395-406.
Louarn, G., Escobar-Gutiérrez, A., Migault, V., Faverjon, L., & Combes, D. (2014). “Virtual grassland”: an individual-based model to deal with grassland community dynamics under fluctuating water and nitrogen availability. The future of european grasslands, 242.



## 1. Getting Started


### 1.1 Prerequisites

To install and use *RIRI5*, you need first to install the dependencies.

*RIRI5* has been tested on Windows.
 

#### 1.1.1 Install the dependencies on Windows 10 64 bit
1) Create a conda environment with miniconda3
    ```bash
    conda create -n myenvname python=3.7 xlrd=2.0.1 numpy=1.20.3 scipy=1.7.3 pandas=1.3.4
    ```

2) Place yourself in the created environment  : `conda activate myenvname`

3) Install *riri5*
    1) Git console :
        ```bash
        git clone -b Develop https://github.com/glouarn/riri5.git
        ```
    2) installation in the conda environment (in folder `riri5`)
        ```bash
        python setup.py develop
        ```


### 1.3 Running

To run a simulation example :

* 1. 
  2. 

## 2. Reading the docs

To build the user and reference guides:


## 3. Testing

The test allows to verify that the model implementation accurately 
represents the developer’s conceptual description of the model and its solution.


## Contact

For any question, send an email to <gaetan.louarn@inrae.fr>.


## Authors

**Gaëtan LOUARN**, **Didier Combes** - see file [AUTHORS](AUTHORS) for details

## License

This project is licensed under the CeCILL-C License - see file [LICENSE](LICENSE) for details
