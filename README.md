# 3D Surface Reconstruction from Point Clouds

A university assignment from the course of Generative AI implementing two methods for reconstructing 3D surfaces from sparse point clouds. The surface is extracted at the zero level-set of an implicit signed distance function (SDF) using the Marching Cubes algorithm.

## Methods

**Naive Geometric Reconstruction** — For each grid point, the signed distance is estimated using the tangent plane of the nearest point in the cloud: `f(p) = n_j · (p - p_j)`.

**Neural Network Reconstruction** — A fully-connected network with 8 layers and skip connections is trained to approximate the SDF directly from 3D coordinates.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        
pip install numpy scikit-image scikit-learn trimesh torch pyglet==1.5.28
```

## Usage

### Naive reconstruction
```bash
python naiveReconstruction.py --input data/bunny-500.pts
python naiveReconstruction.py --input data/bunny-1000.pts
python naiveReconstruction.py --input data/sphere.pts
```

### Neural network — train
```bash
python neuralNetReconstruction.py --input data/bunny-500.pts
```

### Neural network — evaluate (load best saved model and visualize)
```bash
python neuralNetReconstruction.py --input data/bunny-500.pts -e
```

Same commands apply for `bunny-1000.pts` and `sphere.pts`.
