<div align=center>
  <h1>
  Modeling the Distribution of Volcanic Eruptions using Diffusion Models  
  </h1>
  <p>
    <a href=https://mhsung.github.io/kaist-cs492d-fall-2024/ target="_blank"><b>KAIST CS492(D): Diffusion Models and Their Applications (Fall 2024)</b></a><br>
    Course Project
  </p>
</div>

<div align=center>
  <p>
    Instructor: <a href=https://mhsung.github.io target="_blank"><b>Minhyuk Sung</b></a> (mhsung [at] kaist.ac.kr)<br>
    TA: <a href=https://dvelopery0115.github.io target="_blank"><b>Seungwoo Yoo</b></a>  (dreamy1534 [at] kaist.ac.kr)
  </p>
</div>

<div align=center>
   <img src="./assets/images/teaser.png">
   <figcaption>
    A map showing significant volcanic eruptions around the world based on data from the NCEI Significant Volcanic Eruptions Database.
    <i>Source: <a href="https://www.ngdc.noaa.gov/ngdc.html">NOAA National Centers for Environmental Information (NCEI).</a></i>
    </figcaption>
</div>

## Abstract
In this project, you will design diffusion models for modeling the distribution of locations where volcanic eruptions took place. You will use real-world data provided by [NOAA National Centers for Environmental Information (NCEI)](https://www.ngdc.noaa.gov/ngdc.html) to train and evaluate your models.

## Data Specification
The file `data/volcano.tsv` contains training data each consisting of the following fields:
- `Latitude`: Latitude of the location where a volcanic eruption occurred
- `Longitude`: Longitude of the location where a volcanic eruption occurred

## Tasks
Design and implement your own diffusion model for modeling the distribution of volcanic eruptions. Compute the evaluation metrics listed in the next section to assess the performance of your model.

## Evaluation
