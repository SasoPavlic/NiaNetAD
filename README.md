<p align="center"><img src=".github/NiaNetLogo.png" alt="NiaPy" title="NiaNet"/></p>

---
[![PyPI Version](https://img.shields.io/badge/pypi-v1.0.0-blue)](https://pypi.org/project/nianet/)
![PyPI - Python Version](https://img.shields.io/badge/python-3.8-blue)
[![Downloads](https://static.pepy.tech/badge/nianet)](https://pepy.tech/project/nianet)
[![GitHub license](https://img.shields.io/badge/license-MIT-green)](https://github.com/SasoPavlic/NiaNet/blob/main/LICENSE)

### Designing and constructing neural network topologies using nature-inspired algorithms - powered by high performance computer (HPC)

### Description 📝

The proposed method NiaNet attempts to pick hyperparameters and AE architecture that will result in a successful
encoding and decoding (minimal difference between input and output). NiaNet uses the collection of algorithms available
in the library [NiaPy](https://github.com/NiaOrg/NiaPy) to navigate efficiently in waste search-space.

### What it can do? 👀

* **Construct novel AE's architecture** using nature-inspired algorithms.
* * number of solutions = topology shape * layer step * layers * act. func. * epochs * lr * optimizers
* * 3,456,000,000 = 2 * 60 * 60 * 8 * 100 * 100 * 6 
* It can be utilized for **any kind of dataset**, which has **numerical** values.
* Detect anomalies in data

### Installation ✅

Installing NiaNetHPC with pip3:

TODO: Publish it to PyPi

```sh
pip3 install nianet-hpc
```

### Documentation 📘

The purpose of this paper is to get an understanding of the first version of the NiaNet approach (without HPC).

**Annals of Computer Science and Information Systems, Volume 30:**
[NiaNet: A framework for constructing Autoencoder architectures using nature-inspired algorithms](https://www.sasopavlic.com/publication/nianet-a-framework-for-constructing-autoencoder-architectures-using-nat-ure-inspired-algorithms/)

### Examples

Usage examples can be found [here](experiments).

### Getting started 🔨

##### Create your own example:

* TODO Add description for making your own example.

##### Change dataset:

Change the dataset import function as follows:

* TODO Add description for dataloader and config file.

##### Specify the search space:

Set the boundaries of your search space with [autoencoder.py](nianet/autoencoder.py).

The following dimensions can be modified:

* Topology shape (symmetrical, asymmetrical)
* Size of input, hidden and output layers
* Number of hidden layers
* Number of neurons in hidden layers
* Activation functions
* Number of epochs
* Learning rate
* Optimizer

You can run the NiaNet script once your setup is complete.

##### Running NiaNet script with Docker:

```docker build --tag spartan300/nianet:dnnae . ```

```
docker run \
  --name=nianet-dnnae \
  -it \
  -v $(pwd):/app/logs \
  --gpus all spartan300/nianet:dnnae \
  python ./dnn_ae_run.py
```

### HELP ⚠️

**saso.pavlic@student.um.si**

## Acknowledgments 🎓

* NiaNet was developed under the supervision of [doc. dr Iztok Fister ml.](http://www.iztok-jr-fister.eu/)
  at [University of Maribor](https://www.um.si/en/home-page/).

* This code is a fork of [NiaPy](https://github.com/NiaOrg/NiaPy). I am grateful that the authors chose to open-source
  their work for future use.

[comment]: <> (# Cite us)

[comment]: <> (Are you using NiaNet in your project or research? Please cite us!)

[comment]: <> (### Plain format)

[comment]: <> (```)

[comment]: <> (S. Pavlič, I. F. Jr, and S. Karakatič, “NiaNet: A framework for constructing Autoencoder architectures using nature-inspired algorithms,” in Annals of Computer Science and Information Systems, 2022, vol. 30, pp. 109–116. Accessed: Oct. 08, 2022. [Online]. Available: https://annals-csis.org/Volume_30/drp/192.html)

[comment]: <> (```)

[comment]: <> (### Bibtex format)

[comment]: <> (```)

[comment]: <> (    @article{NiaPyJOSS2018,)

[comment]: <> (        author  = {Vrban{\v{c}}i{\v{c}}, Grega and Brezo{\v{c}}nik, Lucija)

[comment]: <> (                  and Mlakar, Uro{\v{s}} and Fister, Du{\v{s}}an and {Fister Jr.}, Iztok},)

[comment]: <> (        title   = {{NiaPy: Python microframework for building nature-inspired algorithms}},)

[comment]: <> (        journal = {{Journal of Open Source Software}},)

[comment]: <> (        year    = {2018},)

[comment]: <> (        volume  = {3},)

[comment]: <> (        issue   = {23},)

[comment]: <> (        issn    = {2475-9066},)

[comment]: <> (        doi     = {10.21105/joss.00613},)

[comment]: <> (        url     = {https://doi.org/10.21105/joss.00613})

[comment]: <> (    })

[comment]: <> (```)

[comment]: <> (### RIS format)

[comment]: <> (```)

[comment]: <> (TY  - CONF)

[comment]: <> (TI  - NiaNet: A framework for constructing Autoencoder architectures using nature-inspired algorithms)

[comment]: <> (AU  - Pavlič, Sašo)

[comment]: <> (AU  - Jr, Iztok Fister)

[comment]: <> (AU  - Karakatič, Sašo)

[comment]: <> (T2  - Proceedings of the 17th Conference on Computer Science and Intelligence Systems)

[comment]: <> (C3  - Annals of Computer Science and Information Systems)

[comment]: <> (DA  - 2022///)

[comment]: <> (PY  - 2022)

[comment]: <> (DP  - annals-csis.org)

[comment]: <> (VL  - 30)

[comment]: <> (SP  - 109)

[comment]: <> (EP  - 116)

[comment]: <> (LA  - en)

[comment]: <> (SN  - 978-83-962423-9-6)

[comment]: <> (ST  - NiaNet)

[comment]: <> (UR  - https://annals-csis.org/Volume_30/drp/192.html)

[comment]: <> (Y2  - 2022/10/08/19:08:20)

[comment]: <> (L1  - https://annals-csis.org/Volume_30/drp/pdf/192.pdf)

[comment]: <> (L2  - https://annals-csis.org/Volume_30/drp/192.html)

[comment]: <> (```)

## License

This package is distributed under the MIT License. This license can be found online
at <http://www.opensource.org/licenses/MIT>.

## Disclaimer

This framework is provided as-is, and there are no guarantees that it fits your purposes or that it is bug-free. Use it
at your own risk!
