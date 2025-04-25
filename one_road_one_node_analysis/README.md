# 🧠 Network Science Project: Road Network Optimization

## 📌 Overview

This project focuses on analyzing and optimizing road networks using network science techniques. We work with GeoJSON data to represent road segments as graphs and apply centrality measures, edge additions, and geospatial preprocessing to identify optimal modifications for enhancing the road network.

Our primary case study is the **Delhi road network**, along with smaller graph datasets for testing and benchmarking.

---

## 📂 Repository Structure

```
.
├── delhi_graph/                     # Graphs and data specific to Delhi
├── experiments/                     # Scripts and test setups
├── small_graph/                     # Smaller test graphs
├── analysis.py                      # Centrality analysis and graph evaluation
├── filtered_output_2_small.geojson # Filtered small graph data
├── filtered_output_delhi.geojson   # Filtered Delhi road data
├── geo_shorting.py                 # Geoshorting logic for small graphs
├── geoshorting_delhi.py            # Delhi network geoshorting
├── optimal_finding.py              # Edge addition optimization
├── README.md                       # Project documentation
```

---

## 🚀 Features

- 🔄 **GeoJSON Road Graph Parsing**: Convert real-world roads into graph structures.
- 🔎 **Centrality Analysis**: Stress, Betweenness and more.
- ➕ **Edge Addition Algorithms**: Identify and add optimal edges to enhance the network.
- 🧭 **Geoshorting**: Smart reduction of path distances in the road graph.
- 📈 **Heuristic Optimization**: Apply rules to lower centrality peaks and improve flow.

---

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd <project-directory>
```

### 2. Install required dependencies

```bash
pip install networkx geopandas shapely tqdm
```

### 3. Run the analysis

```bash
python analysis.py
```

---

## 🧪 Example Scripts

| Script | Purpose |
|--------|---------|
| `geo_shorting.py` | Apply geoshorting to small test graphs |
| `geoshorting_delhi.py` | Optimize paths in Delhi's road network |
| `optimal_finding.py` | Find best edge additions to reduce max stress centrality |
| `analysis.py` | Run centrality analysis and output stats |

---

## 📊 Outputs

- ✅ Filtered and optimized GeoJSON files
- 📋 Console metrics on centrality changes
- 📍 Visual insights into node/edge modifications (add visualization script if needed)

---

## 🙌 Contributing

We welcome contributions! Feel free to fork this repo and submit a pull request. You can also open issues for bugs or suggestions.

---

## 📚 Acknowledgements

- [NetworkX](https://networkx.org/) for graph analysis
- [GeoPandas](https://geopandas.org/) for geospatial data processing
- Road data sourced via **OpenStreetMap** / government datasets

---

## 📌 License

This project is for academic and research purposes only. Please contact us if you'd like to use it commercially or extend it for production-grade tools.
