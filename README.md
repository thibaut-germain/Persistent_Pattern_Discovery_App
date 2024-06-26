
<h1 align="center">Interactive motif discovery in time series with persistent homology</h1>

<div align="center">
<p>
<img alt="GitHub issues" src="https://img.shields.io/github/issues/thibaut-germain/Persistent_Pattern_Discovery_App">
</p>
</div>

<div align="center">

[![Button Icon]](https://persistent-pattern-discovery.onrender.com)

</div>



## Abstract
Time series analysis based on recurrent patterns, also called motifs, has emerged as a powerful approach in various domains. However, uncovering recurrent patterns poses challenges and usually requires expert knowledge. This paper introduces an interactive version of the PersistentPattern algorithm (PEPA), which addresses these challenges by leveraging topological data analysis. PEPA provides a visually intuitive representation of time series, facilitating motif selection without needing expert knowledge. Our work aims to empower data mining and machine learning researchers seeking deeper insights into time series. We provide an overview of the PEPA algorithm and detail its interactive version, concluding with a demonstration of abnormal heartbeat detection.

## Application & algortihm overview 

### Workflow of the PersistentPattern algorithm (PEPA)

<p align="center">
  <img src="src/assets/method_overview.png" alt="drawing" width="1000"/>
  <figcaption>
    <ul>
      <li>Step 1, From time series to graph: Transforms a time series into a graph where nodes are subsequences and edges are weighted with a distance between subsequences. </li>
      <li>Step 2, Graph clustering with persistent homology:  Identifies clusters representing motifs from the persistence diagram and separates them from irrelevant parts of the time series with two thresholds (red lines).</li>
      <li>Step 3: From clusters to motif sets: Merges temporally adjacent subsequences in each cluster to form the variable length motifs.</li>
    </ul>
  </figcaption>
</p>

### Application User Interface

<p align="center">
  <img src="app_screenshot.png" alt="drawing" width="1000"/>
  <figcaption>
   <ul>
     <li>Upper block (red): Associated with step 1, the user uploads a time series, sets parameters related to the graph construction, and runs it.</li>
     <li>Middle left block (green): Associated with step 2, it is the core interactive component of the system. The user can modify the distance function and set the threshold from the resulting persistence diagram.</li>
     <li>Middle right & lower blocks (blue): Associated with step 3, the lower block displays the time series and highlights the discovered motifs. The middle-right block displays motifs individually.</li>
   </ul>
  </figcaption>
</p>

<div align="center">

[![Button Icon]](https://persistent-pattern-discovery.onrender.com)

</div>


## Functionalities

The application can be ran locally with the following command from the root folder
  ```(bash)
  python src/app.py
  ```
Go to the http address given in your terminal by the equivalent message: ```Dash is running on http://127.0.0.1:8050/```.


## Prerequisites


2. All python packages needed are listed in [requirements.txt](https://github.com/thibaut-germain/Persistent-Pattern-Discovery/requirements.txt) file and can be installed simply using the pip command: 

```(bash) 
conda create --name perspa --file requirements.txt
``` 



## Reference

If you use this work, please cite:

<!---------------------------------------------------------------------------->

[Button Icon]: https://img.shields.io/badge/Go_to_application_website-37a779?style=for-the-badge
