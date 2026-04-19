# Generative AI - Coursework & Projects

This repository contains the assignments and the midterm project for the **Generative AI** course at the Technical University of Crete. The curriculum focuses on the mathematical foundations and deep learning architectures used to synthesize 3D geometry and 2D content.

##  Repository Structure

* **Assignment 1: 3D Surface Reconstruction**
    * Implementation of Implicit Functions for 3D shape generation.
    * Geometric reconstruction using Tangent Plane estimation.
    * Neural Implicit Surface learning using a DeepSDF-style MLP with skip connections.
* **Assignment 2: Upcoming**
    * *To be updated.*
* **Midterm Project: Upcoming**
    * *To be updated.*

---

##  Assignment 1: Neural Implicit Surfaces

### Overview
Given a sparse point cloud $P$, we define an implicit function $f(x,y,z)$ to measure the signed distance to the surface. The surface is then extracted at $f(x,y,z) = 0$ using the **Marching Cubes** algorithm.

### Methods Implemented

#### 1. Geometric Tangent Plane Distance
A "naive" approach where the signed distance is calculated based on the distance to the tangent plane of the nearest surface point $p_j$:
$$f(p) = n_j \cdot (p - p_j)$$

#### 2. Deep Neural SDF
A deep MLP trained to approximate the Signed Distance Function (SDF).
* **Architecture**: 8-layer fully-connected network with Weight Normalization and Leaky ReLU.
* **Skip Connections**: Input coordinates $(x, y, z)$ are re-injected at the 4th layer to mitigate vanishing gradients and preserve spatial features.
* **Training**: Optimized using AdamW with a clamped $L_1$ loss function:
    $$L = | \text{clamp}(f_\theta(p_i'), \sigma) - \text{clamp}(s_i, \sigma) |$$
